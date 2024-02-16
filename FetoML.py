import pandas as pd
import tensorflow as tf
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
import joblib
from tensorflow.keras import regularizers
import os
import sys

# User inputs
args = sys.argv
input_file = args[args.index('-input') + 1]
output_name = args[args.index('-output') + 1]
user_model = args[args.index('-model') + 1]
model_list = user_model.split(',')

classifier_list = ['LR', 'SVM', 'RF', 'ET', 'GBM', 'XGB', 'NN']

if len(model_list) == 1:
    if model_list[0] == 'all':
        user_model_selection = classifier_list
    elif model_list[0] == 'recommend':
        user_model_selection = ['ET', 'NN']
    else:
        raise ValueError('Please check the entered argument')

else:
    if 'all' in model_list or 'recommend' in model_list:
        raise ValueError('Please check the "how to use" in Readme file')
    else:
        check_invalid_model = [item for item in model_list if item not in classifier_list]
        if check_invalid_model:
            raise ValueError('Please check the entered argument')
        else:
            user_model_selection = model_list

# Model Code directory definition
cur_dir = os.getcwd()
cur_dir = cur_dir.replace('\\', '/')
model_dir = cur_dir + '/Model Code/'

df_test = pd.read_csv(f'Data/{input_file}', encoding='cp949')
df_test = df_test[['name', 'smiles', 'category']]

test_mols = [Chem.MolFromSmiles(smiles) for smiles in df_test["smiles"]]

none_list = []
for i in range(len(test_mols)):
    if test_mols[i] is None:
        none_list.append(i)
        print('none_list에 추가됨')

reg_idx = 0
for i in none_list:
    del test_mols[i - reg_idx]
    reg_idx += 1

if len(none_list) != 0:
    df_test = df_test.drop(none_list, axis=0)
    df_test = df_test.reset_index(drop=True)

bit_info_list = []  # bit vector의 설명자 리스트 담기
bit_info = {}  # bit vector 설명자
fps = []
b = 0

for a in test_mols:
    fps.append(AllChem.GetMorganFingerprintAsBitVect(a, 3, nBits=2048, bitInfo=bit_info))
    bit_info_list.append(bit_info.copy())  # bit_info 그대로 가져오면 변수가 변해서 리스트 값이 달라지므로 .copy()

arr_list = []
for i in range(len(fps)):
    array = np.zeros((0,), dtype=np.int8)
    arr_list.append(array)
for i in range(len(fps)):
    bit = fps[i]
    DataStructs.ConvertToNumpyArray(bit, arr_list[i])

test_x = np.stack([i.tolist() for i in arr_list])
test_finprt = pd.DataFrame(test_x)

scaler_file_path = os.path.join(model_dir, 'Model/standard_scaler_inform.pkl')
sds_scaler = joblib.load(scaler_file_path)

test_qe = [QED.properties(mol) for mol in test_mols]
test_qe = pd.DataFrame(test_qe)

test_chem = pd.DataFrame()

MW_condition = [test_qe['MW'] < 500]
MW_choice = ['1']
test_chem['MW'] = np.select(MW_condition, MW_choice, default='0')

test_chem['HBOND'] = test_qe['HBA'] + test_qe['HBD']

test_chem['ALOGP'] = test_qe['ALOGP']
test_chem['PSA'] = test_qe['PSA']
test_chem[['HBOND', 'ALOGP', 'PSA']] = sds_scaler.transform(test_chem[['HBOND', 'ALOGP', 'PSA']])

test_input = pd.concat([test_finprt, test_chem, df_test['category']], axis=1)

X_test = test_input.iloc[:, :-1]
Y_test = test_input.iloc[:, -1]

X_test_arr = X_test.values
Y_test_arr = Y_test.values

X_test_nn = np.asarray(X_test).astype('float64')
Y_test_nn = np.asarray(Y_test).astype('float64')

# Logistic Regression
LR_model_file_path = os.path.join(model_dir, 'Model/LogisticRegression.pkl')
LR_model = joblib.load(LR_model_file_path)

# Support Vector Machine
SVC_model_file_path = os.path.join(model_dir, 'Model/SVC.pkl')
SVC_model = joblib.load(SVC_model_file_path)

# Random Forest
RF_model_file_path = os.path.join(model_dir, 'Model/RandomForestClassifier.pkl')
RF_model = joblib.load(RF_model_file_path)

# Extra Trees
ET_model_file_path = os.path.join(model_dir, 'Model/ExtraTreesClassifier.pkl')
ET_model = joblib.load(ET_model_file_path)

# Gradient Boosting Machine
GBM_model_file_path = os.path.join(model_dir, 'Model/GradientBoostingClassifier.pkl')
GBM_model = joblib.load(GBM_model_file_path)

# eXtreme Gradient Boosting
XGB_model_file_path = os.path.join(model_dir, 'Model/XGBClassifier.pkl')
XGB_model = joblib.load(XGB_model_file_path)

# Self-attention-based neural network
input_dim = X_test_nn.shape[1]
initializer = tf.keras.initializers.HeNormal()
regularizer = regularizers.l2(0.001)

# 모델 세부 설정
inputs = tf.keras.layers.Input(shape=(input_dim,))
dense_v = tf.keras.layers.Dense(input_dim, activation=None)(inputs)
attn_score = tf.keras.layers.Softmax(axis=-1)(dense_v)
cal_score = tf.math.multiply(inputs, attn_score)
Dense1 = tf.keras.layers.Dense(116, activation='relu',
                               kernel_initializer=initializer, kernel_regularizer=regularizer)(cal_score)
Dense1_BN = tf.keras.layers.BatchNormalization()(Dense1)
Dense2 = tf.keras.layers.Dense(32, activation='relu',
                               kernel_initializer=initializer, kernel_regularizer=regularizer)(Dense1)
Dense2_BN = tf.keras.layers.BatchNormalization()(Dense2)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(Dense2_BN)

NN_model = tf.keras.Model(inputs=inputs, outputs=outputs)

NN_model.load_weights(os.path.join(model_dir, "Model/attention_best.h5"))

model_thd = {LR_model: 0.227574,
             SVC_model: 0.053477,
             RF_model: 0.114286,
             ET_model: 0.074573,
             GBM_model: 0.424566,
             XGB_model: 0.125577,
             NN_model: 0.002422}


def model_prediction(model_name):
    cls = classifiers.get(model_name)
    if cls is XGB_model:
        pred_z = cls.predict_proba(X_test_arr)[:, 1]
    elif cls is NN_model:
        pred_z = cls.predict(X_test_nn)[:, -1]
    else:
        pred_z = cls.predict_proba(X_test)[:, 1]
    output_probability = pd.Series(pred_z[:])
    predict_result = pred_z.copy()
    predict_result[predict_result >= model_thd.get(cls)] = '1'
    predict_result[predict_result < model_thd.get(cls)] = '0'
    output_binary = pd.Series(predict_result)

    output_df = pd.concat([df_test[['name', 'smiles']], output_probability, output_binary.astype(str)], axis=1)
    output_df.columns = ['name', 'smiles', 'probability', 'output']
    output_df['output'] = output_df['output'].apply(lambda _: str(_))
    output_df.to_csv(f'Results/[{output_name}]_{model_name}_predict_result.csv', encoding='cp949', index=False)

classifiers = {'LR': LR_model,
               'SVM': SVC_model,
               'RF': RF_model,
               'ET': ET_model,
               'GBM': GBM_model,
               'XGB': XGB_model,
               'NN': NN_model}

for a in user_model_selection:
    model_prediction(model_name=a)

print('-*-*' * 20 + '\nIf you checked this message, the result was generated without any issues.\n' + '-*-*' * 20)
