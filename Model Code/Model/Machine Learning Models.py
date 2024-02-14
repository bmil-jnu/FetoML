import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import *
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

warnings.filterwarnings(action='ignore')
sys.path.append('C:/FetoML/Model Code')

df_train = pd.read_csv("../../Data/fetal_toxicity_Train.csv", encoding = 'cp949')
df_train = df_train[['name','smiles','category']]

# mol 형태로 변환
train_mols = [Chem.MolFromSmiles(smiles) for smiles in df_train["smiles"]]

# mol 형태로 변환이 되지 않은 경우, none_list에 담는다
none_list = []
for i in range(len(train_mols)):
    if train_mols[i] is None:
        none_list.append(i)
        print('none_list에 추가됨')

reg_idx = 0
for i in none_list:
    del train_mols[i - reg_idx]
    reg_idx += 1

# none_list가 존재할 경우, 삭제 후 데이터프레임 인덱스 맞춰주기
if len(none_list) != 0:
    df_train = df_train.drop(none_list, axis=0)
    df_train = df_train.reset_index(drop=True)

# fingerprint 생성
bit_info_list = []  # bit vector의 설명자 리스트 담기
bit_info = {}  # bit vector 설명자
fps = []

b = 0

# mol 파일에서 fingerprint Bit Vector 형태로 변환하기
for a in train_mols:
    fps.append(AllChem.GetMorganFingerprintAsBitVect(a, 3, nBits=2048, bitInfo=bit_info))
    bit_info_list.append(bit_info.copy())  # bit_info 그대로 가져오면 변수가 변해서 리스트 값이 달라지므로 .copy()

# array 변환
arr_list = []
for i in range(len(fps)):
    array = np.zeros((0,), dtype=np.int8)
    arr_list.append(array)
for i in range(len(fps)):
    bit = fps[i]
    DataStructs.ConvertToNumpyArray(bit, arr_list[i])

train_x = np.stack([i.tolist() for i in arr_list])
train_finprt = pd.DataFrame(train_x)

# StandardScaler
sds_scaler = StandardScaler()

# molecular physicochemical properties 구하기
train_qe = [QED.properties(mol) for mol in train_mols]
train_qe = pd.DataFrame(train_qe)

train_chem = pd.DataFrame()

# MW 생성
MW_condition = [train_qe['MW'] < 500]
MW_choice = ['1']
train_chem['MW'] = np.select(MW_condition, MW_choice, default = '0')

# HBA+HBD
train_chem['HBOND'] = train_qe['HBA']+train_qe['HBD']

# ALOGP, TPSA, HBA+HBD standard scaling
train_chem['ALOGP'] = train_qe['ALOGP']
train_chem['PSA'] = train_qe['PSA']

train_chem[['HBOND', 'ALOGP', 'PSA']] = sds_scaler.fit_transform(train_chem[['HBOND', 'ALOGP', 'PSA']])

scaler_file_name = 'standard_scaler_inform.pkl'
joblib.dump(sds_scaler, scaler_file_name)

# 생성한 데이터 병합
input_df=pd.concat([train_finprt, train_chem, df_train['category']], axis=1)
X=input_df.iloc[:,:-1]
Y=input_df.iloc[:,-1]
# SMOTE
SMOTE = SMOTE(sampling_strategy='minority', random_state=1472)
X, Y = SMOTE.fit_resample(X, Y)

# LR model
LR_model = LogisticRegression(C=0.34,
                              random_state=1472)
LR_model.fit(X, Y)
joblib.dump(LR_model, f'{LR_model.__class__.__name__}.pkl')

# SVM model
SVC_model = SVC(C=15.92,
                gamma=0.04,
                random_state=1472,
                probability=True)
SVC_model.fit(X, Y)

joblib.dump(SVC_model, f'{SVC_model.__class__.__name__}.pkl')

# RF model
RF_model = RandomForestClassifier(max_depth=68,
                                  max_features=0.15,
                                  min_samples_leaf=1,
                                  min_samples_split=2,
                                  n_estimators=105,
                                  random_state=1472)
RF_model.fit(X, Y)

joblib.dump(RF_model, f'{RF_model.__class__.__name__}.pkl')

# ET model
ET_model = ExtraTreesClassifier(max_depth=82.47,
                                max_features=0.1,
                                min_samples_leaf=1,
                                min_samples_split=11,
                                n_estimators=130,
                                random_state=1472)
ET_model.fit(X, Y)

joblib.dump(ET_model, f'{ET_model.__class__.__name__}.pkl')

# GBM model
GBM_model = GradientBoostingClassifier(learning_rate=0.001,
                                       max_depth=92,
                                       min_samples_leaf=1,
                                       min_samples_split=3,
                                       n_estimators=248,
                                       subsample=0.58,
                                       random_state=1472)
GBM_model.fit(X, Y)

joblib.dump(GBM_model, f'{GBM_model.__class__.__name__}.pkl')

# XGBoost model
X_arr = X.values
Y_arr = Y.values

XGB_model = XGBClassifier(colsample_bytree=0.52,
                          eta=0.02,
                          gamma=0.31,
                          max_depth=13,
                          min_child_weight=1,
                          n_estimators=347,
                          subsample=0.65,
                          eval_metric='logloss',
                          random_state=1472)
XGB_model.fit(X_arr, Y_arr)

joblib.dump(XGB_model, f'{XGB_model.__class__.__name__}.pkl')