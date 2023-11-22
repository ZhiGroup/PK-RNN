# PK-RNN-V
This repo contains the official reference implementation for the paper 
>**PK-RNN-V E: A Deep Learning Model Approach to Vancomycin Therapeutic Drug Monitoring Using Electronic Health Record Data**<br>Nigo Masayuki, Hong Thoai Nga Tran, Ziqian Xie, Han Feng, Laila Rasmy, Miao Hongyu, Degui Zhi
## Requirements
- Python 3.9
- The required Python package can be installed using `pip install -r requirements.txt`
## Overview
The PK-RNN-V model is a recurrent neural network that models the parmacokinetics of vancomycin and predicts the vancomycin concentration.
## Input Specification
The input to the PK-RNN-V is a list of PyTorch tensors:

1. `ContTensor`: All continuous z-score normalized variables of shape (batch, seq, 40).
2. `CatTensor`: All categorical variables of shape (batch, seq, k), where k is the maximum number of medication code of a visit in this batch, other visits are zero padded.
3. `LabelTensor`: Measurements of serum vancomycin of shape (batch, seq). The value is 0 when there is no measurement. **This model requires that the measurements should be taken after the infusion.**
4. `DoseTensor`: Doses of administered vancomycin **in grams** of shape (batch, seq). The value is 0 when there is no dose.
5. `TimeDiffTensor`: Time different between the current event and the next one **in hours** of shape (batch, seq)
6. `VTensor`: Volume of distribution precomputed from weight <img src="https://render.githubusercontent.com/render/math?math=0.0007\times(\text{weight[kg]})[\text{m}^3]"> of shape (batch, seq).
7. `VancoElTensor`: Vancomycin elimination rate precomputed from Matzke equation that relates vancomycin elimination rate to the creatinine clearance <img src="https://render.githubusercontent.com/render/math?math=(0.00083\times(\text{CrCL})%2B0.0044)[\text{hr}^{-1}]"> and the creatinine clearance was calculated from the Cockcroft-Gault equation <img src="https://render.githubusercontent.com/render/math?math=\text{CrCL}=\frac{(140-\text{Age[yr]})\times \text{Weight[kg]}\times(0.85\text{ if Female})}{72\times\text{Serum creatinine[mg/dL]}}\text{[mL/min]}"> then. 
8. `PtList`: Patient ID of shape (batch,), not used in the computation, only for debugging purpose.
9. `LengList`: Sequence length of each patient in the minibatch before padding to the same length of shape (batch,), not used in the computation, only for debugging purpose.

Following is the table of 41 z-score normalized continuous variables used in the model and their mean and standard deviation in our dataset:

| Index | Description | Mean       | Std        |
|-------|---|------------|------------|
|1|Time Difference| 14.797080  | 303.458232 |
|2| Age| 58.239251  | 16.936469  |
|3| Serum Calcium Level| 8.501884   | 0.675741   |
|4| Serum Carbon dioxide level| 25.649442  | 4.106355   |
|5| Serum Chloride level| 105.078446 | 5.869722   |
|6| Serum Phosphorus level| 3.222309   | 1.086376   |
|7| Serum Potassium level| 3.983608   | 0.556475   |
|8| Serum Sodium level| 138.639875 | 4.556864   |
|9| Serum Albumin level| 2.628429   | 0.652720   |
|10| Serum total bilirubin level| 0.861545   | 1.358118   |
|11| Serum Creatinine level| 1.314668   | 1.418099   |
|12| Serum Blood Urea Nitrogen level| 22.156025  | 17.348442  |
|13| Estimated Glomerular Filtration Rate| 77.284150  | 36.319045  |
|14| Serum Total Protein level| 6.877492   | 1.056723   |
|15| Serum Glucose level| 153.041354 | 70.350104  |
|16| Hematocrit level| 31.762647  | 6.468392   |
|17| Hemoglobin level| 10.490741  | 2.201154   |
|18| Serum Magnesium level| 2.002053   | 0.377566   |
|19| Body Temperature| 36.916618  | 0.544462   |
|20| Height| 5.602557   | 0.549278   |
|21| Weight| 87.575200  | 30.519849  |
|22| Heart rate| 85.363223  | 15.410158  |
|23| Respiratory rate| 18.907366  | 3.698564   |
|24| Systolic Blood Pressure| 127.447943 | 19.798996  |
|25| Diastolic Blood Pressure| 68.435240  | 13.470637  |
|26| SpO2 | 97.008675  | 2.478909   |
|27| O2 Flow| 5.266053   | 6.763950   |
|28| Blood Lymphocytes Percentage| 16.262886  | 11.432474  |
|29| Blood Lymphocytes Count| 1.543559   | 4.659094   |
|30| Blood Monocytes Percentage| 8.324932   | 4.154349   |
|31| Blood Monocytes Count| 0.837809   | 0.650876   |
|32| Blood Eosinophils Percentage| 2.314697   | 2.975779   |
|33| Blood Eosinophils Count| 0.245451   | 0.265149   |
|34| Blood Neutrophils Count| 8.330551   | 5.609378   |
|35| Nucleated Red Blood Cell (%)| 0.628276   | 5.774708   |
|36| Platelets counts| 273.986166 | 138.682088 |
|37| Red Blood Cell Counts| 3.652529   | 0.761418   |
|38| Blood Basophils Percentage| 0.566043   | 0.522798   |
|39| Blood Basophils Counts| 0.091768   | 0.067752   |
|40| White Blood Cell| 10.865301  | 7.446581   |

The catogorical variables consist of medication code. The medication code are first mapped to integers and then transformed into embeddings, the mapping between medication code used in this study and integers is stored in `code_dict.pkl`.
## VTDM
This code also contains a reference implementation of a 2-compartment Bayesian vancomycin therapeutic drug monitoring model (VTDM). VTDM takes the following input:
1. `dose_tensor`: Similar to `DoseTensor` in PK-RNN-V, but just 1-D.
2. `tdiff_tensor`: Similar to `TimeDiffTensor` in PK-RNN-V, but just 1-D.
3. `ccl_tensor`: Creatinine clearance computed using the Cockcroft-Gault equation, 1-D.
4. `vanco_level_tensor`: Serum vancomycin levels, 1-D, not used during prediction but only the first measurement is used in the inference in this implementation.

Usage:
```python
from model import VTDM

vtdm = VTDM()
vtdm.inference(dose_tensor, tdiff_tensor, ccl_tensor, vanco_level_tensor)
prediction = vtdm.predict(dose_tensor, tdiff_tensor, ccl_tensor)
```
