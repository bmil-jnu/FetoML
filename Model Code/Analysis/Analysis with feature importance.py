import sys
import warnings
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from sklearn.inspection import permutation_importance

warnings.filterwarnings(action='ignore')
sys.path.append('C:/FetoML/Model Code')

df_test = pd.read_csv("../../Data/fetal_toxicity_Test.csv", encoding = 'cp949')
df_test = df_test[['name','smiles','category']]

# mol 형태로 변환
test_mols = [Chem.MolFromSmiles(smiles) for smiles in df_test["smiles"]]

# mol 형태로 변환이 되지 않은 경우, none_list에 담는다
none_list = []
for i in range(len(test_mols)):
    if test_mols[i] is None:
        none_list.append(i)
        print('none_list에 추가됨')

reg_idx = 0
for i in none_list:
    del test_mols[i - reg_idx]
    reg_idx += 1

# none_list가 존재할 경우, 삭제 후 데이터프레임 인덱스 맞춰주기
if len(none_list) != 0:
    df_test = df_test.drop(none_list, axis=0)
    df_test = df_test.reset_index(drop=True)

# fingerprint 생성
bit_info_list = []  # bit vector의 설명자 리스트 담기
bit_info = {}  # bit vector 설명자
fps = []

b = 0

# mol 파일에서 fingerprint Bit Vector 형태로 변환하기
for a in test_mols:
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

test_x = np.stack([i.tolist() for i in arr_list])
test_finprt = pd.DataFrame(test_x)


# StandardScaler
scaler_file_path = '../Model/standard_scaler_inform.pkl'
sds_scaler = joblib.load(scaler_file_path)

# molecular physicochemical properties 구하기
test_qe = [QED.properties(mol) for mol in test_mols]
test_qe = pd.DataFrame(test_qe)

test_chem = pd.DataFrame()

# MW 생성
MW_condition = [test_qe['MW'] < 500]
MW_choice = ['1']
test_chem['MW'] = np.select(MW_condition, MW_choice, default = '0')

# HBA+HBD
test_chem['HBOND'] = test_qe['HBA']+test_qe['HBD']

# ALOGP, TPSA, HBA+HBD standard scaling
test_chem['ALOGP'] = test_qe['ALOGP']
test_chem['PSA'] = test_qe['PSA']

test_chem[['HBOND', 'ALOGP', 'PSA']] = sds_scaler.transform(test_chem[['HBOND', 'ALOGP', 'PSA']])

# 생성한 데이터 병합
test_input=pd.concat([test_finprt, test_chem, df_test['category']], axis=1)

X_test = test_input.iloc[:, :-1]
Y_test = test_input.iloc[:, -1]

X_test_arr = X_test.values
Y_test_arr = Y_test.values

# Logistic Regression
LR_model_file_path = '../Model/LogisticRegression.pkl'
LR_model = joblib.load(LR_model_file_path)

# Support Vector Machine
SVC_model_file_path = '../Model/SVC.pkl'
SVC_model = joblib.load(SVC_model_file_path)

# Random Forest
RF_model_file_path = '../Model/RandomForestClassifier.pkl'
RF_model = joblib.load(RF_model_file_path)

# Extra Trees
ET_model_file_path = '../Model/ExtraTreesClassifier.pkl'
ET_model = joblib.load(ET_model_file_path)

# Gradient Boosting Machine
GBM_model_file_path = '../Model/GradientBoostingClassifier.pkl'
GBM_model = joblib.load(GBM_model_file_path)

# eXtreme Gradient Boosting
XGB_model_file_path = '../Model/XGBClassifier.pkl'
XGB_model = joblib.load(XGB_model_file_path)

def make_feature_importances(model_list, X, Y, n_repeats=30, random_state=0, scoring='roc_auc', rank=15):
    df_feature_importance = pd.DataFrame()
    list_important_features = []
    significant_features = []
    for model in model_list:
        if model.__class__.__name__ is 'XGBClassifier':
            X_arr = X.values
            Y_arr = Y.values
            Permutator = permutation_importance(model, X_arr, Y_arr,
                                                n_repeats=n_repeats,
                                                random_state=random_state,
                                                scoring=scoring)
        else:
            Permutator = permutation_importance(model, X, Y,
                                                n_repeats=n_repeats,
                                                random_state=random_state,
                                                scoring=scoring)

        feature_importance = {}
        valid_count = 0
        for i in Permutator.importances_mean.argsort()[::-1]:
            if Permutator.importances_mean[i] - 2 * Permutator.importances_std[i] > 0:
                feature_importance[X.columns[i]] = Permutator.importances_mean[i]
        print(f'{model.__class__.__name__}: {Permutator.importances_mean.argsort()[::-1][:rank]}')

        df_FI = pd.DataFrame.from_dict(feature_importance, orient='index', columns=[model.__class__.__name__])
        df_FI = df_FI.apply(lambda x: (x / x.sum()) * 100)  # %로 변환
        df_feature_importance = pd.concat([df_feature_importance, df_FI], axis=1)

        df_FI = df_FI.sort_values(by=model.__class__.__name__, ascending=False)
        top_rank_features = list(df_FI.iloc[:rank].index)

        print(df_FI)
        print(top_rank_features)

        list_important_features = list_important_features + top_rank_features

        print(list_important_features)

    important_features = Counter(list_important_features)

    for key, value in important_features.items():
        if value > 1:
            significant_features.append(key)

    df_specific_FI = df_feature_importance.loc[significant_features]
    df_specific_FI = df_specific_FI.fillna(0)

    return df_feature_importance, df_specific_FI, important_features

model_list = [LR_model, SVC_model, RF_model, ET_model, GBM_model, XGB_model]

df_feature_importance, df_specific_FI, count_FI = make_feature_importances(model_list,
                                                                X_test,
                                                                Y_test,
                                                                n_repeats=30,
                                                                random_state=1472,
                                                                scoring='roc_auc',
                                                                rank=15)


df_specific_FI.columns = ['LR', 'SVM', 'RF', 'ET', 'GBM', 'XGBoost']

title_size = 20
tick_size = 12
label_size = 14

widths = [0.75, 1.25, 6, 3]

fig = plt.figure(figsize=(8, 0.6 * len(df_specific_FI.index)))

spec = fig.add_gridspec(ncols=4, nrows=1, width_ratios=widths)

axs = {}
for i in range(len(widths)):
    axs[i] = fig.add_subplot(spec[i // len(widths), i % len(widths)])

sns.heatmap(df_specific_FI, ax=axs[2], cbar=False, cmap=sns.light_palette("#79C", as_cmap=True),
            linewidth=0.5, square=False, robust=True)
axs[2].set_xlabel('Model', fontsize=label_size)
axs[2].set_ylabel('Feature name', fontsize=label_size)
axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=60, fontsize=tick_size)
axs[2].set_yticklabels(axs[2].get_yticklabels(), rotation=0, fontsize=tick_size)

axs[3].barh(np.arange(0.5, len(df_specific_FI.index)), np.mean(df_specific_FI, axis=1), color='cornflowerblue')
axs[3].set_ylim(axs[2].get_ylim())
axs[3].set_xlabel('Avg. %', fontsize=label_size)
axs[3].set_yticklabels([])
axs[3].tick_params(labelsize=tick_size, length=0)
axs[3].spines["bottom"].set_visible(False)
axs[3].spines["top"].set_visible(False)
axs[3].spines["right"].set_visible(False)

cbar = plt.colorbar(axs[2].get_children()[0], cax=axs[0], orientation='vertical')
axs[0].set_ylabel('')
axs[0].tick_params(labelsize=tick_size, length=0, labelleft=True, labelright=False)

axs[1].axis("off")

fig.suptitle('% of Feature Importance by Model', fontsize=title_size)
plt.tight_layout()
fig.subplots_adjust(wspace=0.4)

fig.savefig('feature importance.png', dpi=600)

print(df_specific_FI)