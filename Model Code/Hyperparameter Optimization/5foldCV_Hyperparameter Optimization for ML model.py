import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from rdkit.Chem import QED
from imblearn.pipeline import make_pipeline
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

sys.path.append('C:/FetoML/Model Code')

df_train = pd.read_csv("../../Data/fetal_toxicity_Train.csv", encoding='cp949')
df_train = df_train[['name', 'smiles', 'category']]

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

train_chem['MW'] = np.select(MW_condition, MW_choice, default='0')

# HBA+HBD

train_chem['HBOND'] = train_qe['HBA'] + train_qe['HBD']

# ALOGP, TPSA, HBA+HBD standard scaling

train_chem['ALOGP'] = train_qe['ALOGP']
train_chem['PSA'] = train_qe['PSA']

train_chem[['HBOND', 'ALOGP', 'PSA']] = sds_scaler.fit_transform(train_chem[['HBOND', 'ALOGP', 'PSA']])

scaler_file_name = 'standard_scaler_inform.pkl'
joblib.dump(sds_scaler, scaler_file_name)


# 생성한 데이터 병합

input_df = pd.concat([train_finprt, train_chem, df_train['category']], axis=1)

X = input_df.iloc[:, :-1]
Y = input_df.iloc[:, -1]

X_arr = X.values
Y_arr = Y.values



# Stratified K fold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1472)

# 로지스틱 리그레션 함수 정의
def lr_bo(expC):
    lr_params = {
        'C': 10 ** expC,
        'random_state': 1472
    }

    LR_model = make_pipeline(SMOTE(sampling_strategy='minority', random_state=1472),
                             LogisticRegression(**lr_params))
    cval = cross_val_score(LR_model, X, Y, scoring="accuracy", cv=skf)

    return cval.mean()


# 하이퍼 파라미터 범위 설정
lr_bounds = {
    'expC': (-3, 2)
}

# Bayesian Hyperparameter Optimization

BO_lr = BayesianOptimization(f=lr_bo, pbounds=lr_bounds, random_state=1472)

BO_lr.maximize(n_iter=200)


# SVM 함수 정의
def svm_bo(expC, expGamma):
    svm_params = {
        'C': 10 ** expC,
        'gamma': 10 ** expGamma,
        'random_state': 1472
    }

    SVC_model = make_pipeline(SMOTE(sampling_strategy='minority', random_state=1472),
                              SVC(**svm_params))
    cval = cross_val_score(SVC_model, X, Y, scoring="accuracy", cv=skf)

    return cval.mean()


# 하이퍼 파라미터 범위 설정
svm_bounds = {
    'expC': (-3, 2),
    'expGamma': (-4, -1)
}

# Bayesian Hyperparameter Optimization

BO_svm = BayesianOptimization(f=svm_bo, pbounds=svm_bounds, random_state=1472)

BO_svm.maximize(n_iter=200)

# RF 함수 정의
def rf_bo(n_estimators, min_samples_split, max_features, max_depth, min_samples_leaf):
    rf_params = {
        'n_estimators': int(n_estimators),
        'min_samples_split': int(min_samples_split),
        'max_features': max(min(max_features, 0.999), 1e-3),
        'max_depth': int(max_depth),
        'min_samples_leaf': int(min_samples_leaf),
        'random_state': 1472
    }

    RF_model = make_pipeline(SMOTE(sampling_strategy='minority', random_state=1472),
                             RandomForestClassifier(**rf_params))
    cval = cross_val_score(RF_model, X, Y, scoring="accuracy", cv=skf)

    return cval.mean()


# 하이퍼 파라미터 범위 설정
rf_bounds = {
    'n_estimators': (10, 250),
    'min_samples_split': (2, 25),
    'max_features': (0.1, 0.999),
    'max_depth': (30, 100),
    'min_samples_leaf': (1, 10)
}

# Bayesian Hyperparameter Optimization
BO_rf = BayesianOptimization(f=rf_bo, pbounds=rf_bounds, random_state=1472)

BO_rf.maximize(n_iter=200)

def et_bo(n_estimators, min_samples_split, max_features, max_depth, min_samples_leaf):
    et_params = {
        'n_estimators': int(n_estimators),
        'min_samples_split': int(min_samples_split),
        'max_features': max(min(max_features, 0.999), 1e-3),
        'max_depth': int(max_depth),
        'min_samples_leaf': int(min_samples_leaf),
        'random_state': 1472
    }

    ET_model = make_pipeline(SMOTE(sampling_strategy='minority', random_state=1472),
                             ExtraTreesClassifier(**et_params))
    cval = cross_val_score(ET_model, X, Y, scoring="accuracy", cv=skf)

    return cval.mean()


# 하이퍼 파라미터 범위 설정
et_bounds = {
    'n_estimators': (10, 250),
    'min_samples_split': (10, 50),
    'max_features': (0.1, 0.999),
    'max_depth': (20, 100),
    'min_samples_leaf': (1, 20)
}

# Bayesian Hyperparameter Optimization
BO_et = BayesianOptimization(f=et_bo, pbounds=et_bounds, random_state=1472)
BO_et.maximize(n_iter=200)

# GBM 함수 정의
def gbm_bo(learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, subsample):
    gbm_params = {
        'learning_rate': 10 ** learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'subsample': subsample,
        'random_state': 1472
    }

    GBM_model = make_pipeline(SMOTE(sampling_strategy='minority', random_state=1472),
                              GradientBoostingClassifier(**gbm_params))
    cval = cross_val_score(GBM_model, X, Y, scoring="accuracy", cv=skf)

    return cval.mean()


# 하이퍼 파라미터 범위 설정
gbm_bounds = {
    'learning_rate': (-1, -3),
    'n_estimators': (200, 500),
    'max_depth': (1, 100),
    'min_samples_split': (2, 25),
    'min_samples_leaf': (1, 25),
    'subsample': (0, 1)
}

# Bayesian Hyperparameter Optimization
BO_gbm = BayesianOptimization(f=gbm_bo, pbounds=gbm_bounds, random_state=1472)
BO_gbm.maximize(n_iter=200)

# X 와 Y 데이터프레임 -> 어레이 형태로
X_arr = X.values
Y_arr = Y.values

# XGB 함수 정의
def xgb_bo(eta, n_estimators, min_child_weight, gamma, max_depth, subsample, colsample_bytree):
    xgb_params = {
        'eta': 10 ** (eta),
        'n_estimators': int(n_estimators),
        'min_child_weight': int(min_child_weight),
        'gamma': gamma,
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': -1,
        'seed': 1472
    }

    XGB_model = make_pipeline(SMOTE(sampling_strategy='minority', random_state=1472),
                              XGBClassifier(**xgb_params))
    cval = cross_val_score(XGB_model, X_arr, Y_arr, scoring="accuracy", cv=skf)

    return cval.mean()


# 하이퍼 파라미터 범위 설정
xgb_bounds = {'eta': (-1, -3),
              'n_estimators': (300, 700),
              'min_child_weight': (1, 2),
              'gamma': (0, 2),
              'max_depth': (5, 15),
              'subsample': (0.5, 1),
              'colsample_bytree': (0.5, 1)
              }

# Bayesian Hyperparameter Optimization
BO_xgb = BayesianOptimization(f=xgb_bo, pbounds=xgb_bounds, random_state=1472)
BO_xgb.maximize(n_iter=50)

# Logistic Regression Best Parameters
lr_max_params = BO_lr.max['params']
lr_max_params['C'] = 10 ** (lr_max_params['expC'])
print('=' * 50)
print(f'Logistic Regression Best Parameters: {lr_max_params}')
print('=' * 50)

# Support Vector Machine Best Parameters
svm_max_params = BO_svm.max['params']
svm_max_params['C'] = 10 ** (svm_max_params['expC'])
svm_max_params['Gamma'] = 10 ** (svm_max_params['expGamma'])
print('=' * 50)
print(f'Support Vector Machine Best Parameters: {svm_max_params}')
print('=' * 50)

# Random Forest Best Parameters
rf_max_params = BO_rf.max['params']
rf_max_params['n_estimators'] = int(rf_max_params['n_estimators'])
rf_max_params['min_samples_split'] = int(rf_max_params['min_samples_split'])
rf_max_params['max_features'] = max(min(rf_max_params['max_features'], 0.999), 1e-3)
rf_max_params['max_depth'] = int(rf_max_params['max_depth'])
rf_max_params['min_samples_leaf'] = int(rf_max_params['min_samples_leaf'])
print('=' * 50)
print(f'Random Forest Best Parameters: {rf_max_params}')
print('=' * 50)

# Extra Tree Best Parameters
et_max_params = BO_et.max['params']
et_max_params['n_estimators'] = int(et_max_params['n_estimators'])
et_max_params['min_samples_split'] = int(et_max_params['min_samples_split'])
et_max_params['max_features'] = max(min(et_max_params['max_features'], 0.999), 1e-3)
et_max_params['min_samples_leaf'] = int(et_max_params['min_samples_leaf'])
print('=' * 50)
print(f'Extra Tree Best Parameters: {et_max_params}')
print('=' * 50)

# Gradient Boosting Machine Best Parameters
gbm_max_params = BO_gbm.max['params']
gbm_max_params['learning_rate'] = 10 ** (gbm_max_params['learning_rate'])
gbm_max_params['n_estimators'] = int(gbm_max_params['n_estimators'])
gbm_max_params['max_depth'] = int(gbm_max_params['max_depth'])
gbm_max_params['min_samples_split'] = int(gbm_max_params['min_samples_split'])
gbm_max_params['min_samples_leaf'] = int(gbm_max_params['min_samples_leaf'])
print('=' * 50)
print(f'Gradient Boosting Machine Best Parameters: {gbm_max_params}')
print('=' * 50)

# eXtreme Gradient Boosting Best Parameters
xgb_max_params = BO_xgb.max['params']
xgb_max_params['eta'] = 10 ** (xgb_max_params['eta'])
xgb_max_params['n_estimators'] = int(xgb_max_params['n_estimators'])
xgb_max_params['min_child_weight'] = int(xgb_max_params['min_child_weight'])
xgb_max_params['max_depth'] = int(xgb_max_params['max_depth'])
print('=' * 50)
print(f'eXtreme Gradient Boosting Best Parameters: {xgb_max_params}')
print('=' * 50)
