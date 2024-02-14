import sys
import warnings
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
import joblib
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve

sys.path.append('C:/FetoML/Model Code')
warnings.filterwarnings(action='ignore')

df_test = pd.read_csv("../../Data/fetal_toxicity_Test.csv", encoding='cp949')
df_test = df_test[['name', 'smiles', 'category']]

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
test_chem['MW'] = np.select(MW_condition, MW_choice, default='0')

# HBA+HBD
test_chem['HBOND'] = test_qe['HBA'] + test_qe['HBD']

# ALOGP, TPSA, HBA+HBD standard scaling
test_chem['ALOGP'] = test_qe['ALOGP']
test_chem['PSA'] = test_qe['PSA']
test_chem[['HBOND', 'ALOGP', 'PSA']] = sds_scaler.transform(test_chem[['HBOND', 'ALOGP', 'PSA']])
test_chem

# 생성한 데이터 병합
test_input = pd.concat([test_finprt, test_chem, df_test['category']], axis=1)

X_test = test_input.iloc[:, :-1]
Y_test = test_input.iloc[:, -1]

X_test_arr = X_test.values
Y_test_arr = Y_test.values

X_test_nn = np.asarray(X_test).astype('float64')
Y_test_nn = np.asarray(Y_test).astype('float64')

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

# Self-attention-based neural network
input_dim = X_test_nn.shape[1]
initializer = tf.keras.initializers.HeNormal()
regularizer = regularizers.l2(0.001)

# 모델 세부 설정
inputs = tf.keras.layers.Input(shape=(input_dim,))
dense_v = tf.keras.layers.Dense(input_dim, activation = None)(inputs)
attn_score = tf.keras.layers.Softmax(axis = -1)(dense_v)
cal_score = tf.math.multiply(inputs, attn_score)
Dense1 = tf.keras.layers.Dense(116, activation = 'relu',
                          kernel_initializer = initializer, kernel_regularizer=regularizer)(cal_score)
Dense1_BN = tf.keras.layers.BatchNormalization()(Dense1)
Dense2 = tf.keras.layers.Dense(32, activation = 'relu',
                          kernel_initializer = initializer, kernel_regularizer=regularizer)(Dense1)
Dense2_BN = tf.keras.layers.BatchNormalization()(Dense2)
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(Dense2_BN)

NN_model = tf.keras.Model(inputs=inputs, outputs=outputs)

NN_model.load_weights("../Model/attention_best.h5")

# ROC curve
classifiers = [LR_model,
               SVC_model,
               RF_model,
               ET_model,
               GBM_model,
               XGB_model,
               NN_model]

results_ROC = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

for cls in classifiers:
    if cls is XGB_model:
        pred_z = cls.predict_proba(X_test_arr)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test_arr, pred_z)
        roc_auc = roc_auc_score(Y_test_arr, pred_z)
    elif cls is NN_model:
        pred_z = cls.predict(X_test_nn)
        fpr, tpr, _ = roc_curve(Y_test_nn, pred_z)
        roc_auc = roc_auc_score(Y_test_nn, pred_z)
    else:
        pred_z = cls.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, pred_z)
        roc_auc = roc_auc_score(Y_test, pred_z)

    if cls is NN_model:
        results_ROC = results_ROC.append({'classifiers': 'Self-attention-based neural network',
                                          'fpr': fpr,
                                          'tpr': tpr,
                                          'auc': roc_auc}, ignore_index=True)
    else:
        results_ROC = results_ROC.append({'classifiers': cls.__class__.__name__,
                                          'fpr': fpr,
                                          'tpr': tpr,
                                          'auc': roc_auc}, ignore_index=True)

results_ROC.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8, 6))

for i in results_ROC.index:
    plt.plot(results_ROC.loc[i]['fpr'],
             results_ROC.loc[i]['tpr'],
             label=f'{i}, AUC={results_ROC.loc[i]["auc"]:.3f}')

plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.xlim(0.0, 1.01)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.ylim(0.0, 1.03)

plt.title('ROC Curve of Machine Learning Methods', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')
plt.show

# PR curve
classifiers = [LR_model,
               SVC_model,
               RF_model,
               ET_model,
               GBM_model,
               XGB_model,
               NN_model]

results_PR = pd.DataFrame(columns=['classifiers', 'recall', 'precision', 'auc'])

for cls in classifiers:
    if cls is XGB_model:
        pred_z = cls.predict_proba(X_test_arr)
        precision, recall, thresholds = precision_recall_curve(Y_test_arr, pred_z[:, 1])
        pr_auc = auc(recall, precision)
    elif cls is NN_model:
        pred_z = cls.predict(X_test_nn)
        precision, recall, thresholds = precision_recall_curve(Y_test_nn, pred_z)
        pr_auc = auc(recall, precision)
    else:
        pred_z = cls.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(Y_test, pred_z[:, 1])
        pr_auc = auc(recall, precision)

    if cls is NN_model:
        results_PR = results_PR.append({'classifiers': 'Self-attention-based neural network',
                                        'recall': recall,
                                        'precision': precision,
                                        'auc': pr_auc}, ignore_index=True)
    else:
        results_PR = results_PR.append({'classifiers': cls.__class__.__name__,
                                        'recall': recall,
                                        'precision': precision,
                                        'auc': pr_auc}, ignore_index=True)

results_PR.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8, 6))

for i in results_PR.index:
    plt.plot(results_PR.loc[i]['recall'],
             results_PR.loc[i]['precision'],
             label=f'{i}, AUC={results_PR.loc[i]["auc"]:.3f}')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlim(0.0, 1.01)
plt.xlabel("Recall", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylim(0.0, 1.03)
plt.ylabel("Precision", fontsize=15)

plt.title('PR Curve of Machine Learning Methods', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower left')
plt.show

# Quantitative performance evaluation
def perform_metrics(model, X, Y_true):
    pred_z = model.predict_proba(X)

    precision, recall, thresholds = precision_recall_curve(Y_test_arr, pred_z[:, 1])
    F1 = 2 * (precision * recall) / (precision + recall)
    th_precision = pd.DataFrame(precision, columns=['precision'])
    th_recall = pd.DataFrame(recall, columns=['recall'])
    th_threshold = pd.DataFrame(thresholds, columns=['threshold'])
    th_F1 = pd.DataFrame(F1, columns=['F1 Score'])
    th = pd.concat([th_precision, th_recall, th_threshold, th_F1], axis=1)

    bestloc = th['F1 Score'].argmax()
    best_threshold = threshold = th['threshold'].loc[bestloc]

    print(f'Best threshold: {best_threshold}')

    pred_z[pred_z >= best_threshold] = 1
    pred_z[pred_z < best_threshold] = 0

    print(f'Model: {model}')
    print(f'Accuracy: {accuracy_score(Y_true, pred_z[:, 1]):.4f}')
    print(f'Precision: {precision_score(Y_true, pred_z[:, 1]):.4f}')
    print(f'Recall: {recall_score(Y_true, pred_z[:, 1]):.4f}')
    print(f'F1 score: {f1_score(Y_true, pred_z[:, 1]):.4f}')

# False positive analysis
def FP_predicted_score(clf, X, Y):
    if clf.__class__.__name__ == 'XGBClassifier':
        pred_z = clf.predict_proba(X.values)
        precision, recall, thresholds = precision_recall_curve(Y_test_arr, pred_z[:, 1])
    elif clf.__class__.__name__ == 'Functional':
        X = np.asarray(X).astype('float64')
        pred_z = clf.predict(X)
        precision, recall, thresholds = precision_recall_curve(Y_test_arr, pred_z)
    else:
        pred_z = clf.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(Y_test, pred_z[:, 1])

    F1 = 2 * (precision * recall) / (precision + recall)
    th_precision = pd.DataFrame(precision, columns=['precision'])
    th_recall = pd.DataFrame(recall, columns=['recall'])
    th_threshold = pd.DataFrame(thresholds, columns=['threshold'])
    th_F1 = pd.DataFrame(F1, columns=['F1 Score'])
    th = pd.concat([th_precision, th_recall, th_threshold, th_F1], axis=1)

    bestloc = th['F1 Score'].argmax()
    best_threshold = threshold = th['threshold'].loc[bestloc]

    pred_z = pd.Series(pred_z[:, -1])
    clf_predict_matrix = pd.concat([pred_z, Y], axis=1)
    clf_FP_top10 = clf_predict_matrix[(clf_predict_matrix['category'] == 0)
                                      & (clf_predict_matrix[0] > 0)
                                      ].sort_values(by=0, ascending=False).head(10)
    clf_FP_top10_index = list(clf_FP_top10.index)
    clf_FP_top10 = pd.concat([clf_FP_top10, df_test['name'].iloc[clf_FP_top10_index]], axis=1)
    clf_FP_top10 = clf_FP_top10[['name', 0, 'category']]
    clf_FP_top10.columns = ['name', f'{clf.__class__.__name__} prediction score', 'category']
    clf_FP_top10.to_csv(f'{clf.__class__.__name__}.csv')

    print(best_threshold)

    return clf_FP_top10

