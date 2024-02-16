import os
import sys
import xml.etree.ElementTree as ET
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import *
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import AUC
from xgboost.sklearn import XGBClassifier

args = sys.argv
train_file = args[args.index('-train') + 1]
test_file = args[args.index('-test') + 1]
project_name = args[args.index('-name') + 1]
output_name = args[args.index('-output') + 1]

cur_dir = os.getcwd()
cur_dir = cur_dir.replace('\\', '/')

# Create the project directory

project_dir = cur_dir + f'/{project_name}'
project_models = project_dir + '/Models/'
project_result = project_dir + '/Model Results/'

if not os.path.exists(project_dir):
    os.makedirs(project_dir)
    os.makedirs(project_models)
    os.makedirs(project_result)
    print(f"Directory created at {project_dir}")
else:
    print(f"Directory already exists at {project_dir}")

model_dir = cur_dir + '/Model Code/'

df_train = pd.read_csv(f"Data/{train_file}", encoding='cp949')
df_train = df_train[['name', 'smiles', 'category']]

# Molecular descriptor preprocessing
train_mols = [Chem.MolFromSmiles(smiles) for smiles in df_train["smiles"]]
none_list = []
for i in range(len(train_mols)):
    if train_mols[i] is None:
        none_list.append(i)
        print('Appended in none_list')

reg_idx = 0
for i in none_list:
    del train_mols[i - reg_idx]
    reg_idx += 1

if len(none_list) != 0:
    df_train = df_train.drop(none_list, axis=0)
    df_train = df_train.reset_index(drop=True)

bit_info_list = []
bit_info = {}
fps = []
b = 0

for a in train_mols:
    fps.append(AllChem.GetMorganFingerprintAsBitVect(a, 3, nBits=2048, bitInfo=bit_info))
    bit_info_list.append(bit_info.copy())
arr_list = []

for i in range(len(fps)):
    array = np.zeros((0,), dtype=np.int8)
    arr_list.append(array)
for i in range(len(fps)):
    bit = fps[i]
    DataStructs.ConvertToNumpyArray(bit, arr_list[i])

train_x = np.stack([i.tolist() for i in arr_list])
train_finprt = pd.DataFrame(train_x)

train_qe = [QED.properties(mol) for mol in train_mols]
train_qe = pd.DataFrame(train_qe)
train_chem = pd.DataFrame()

MW_condition = [train_qe['MW'] < 500]
MW_choice = ['1']
train_chem['MW'] = np.select(MW_condition, MW_choice, default='0')
train_chem['HBOND'] = train_qe['HBA'] + train_qe['HBD']
train_chem['ALOGP'] = train_qe['ALOGP']
train_chem['PSA'] = train_qe['PSA']

# StandardScaler
sds_scaler = StandardScaler()
train_chem[['HBOND', 'ALOGP', 'PSA']] = sds_scaler.fit_transform(train_chem[['HBOND', 'ALOGP', 'PSA']])

scaler_file_path = os.path.join(project_models, 'standard_scaler_inform.pkl')
joblib.dump(sds_scaler, scaler_file_path)

input_df = pd.concat([train_finprt, train_chem, df_train['category']], axis=1)
X = input_df.iloc[:, :-1]
Y = input_df.iloc[:, -1]

# For NN models

X_nn = X.copy()
Y_nn = Y.copy()

X_nn = np.asarray(X_nn).astype('float64')
Y_nn = np.asarray(Y_nn).astype('float64')

X_nn_train, X_nn_valid, Y_nn_train, Y_nn_valid = train_test_split(X_nn, Y_nn, test_size=0.2, random_state=1472)

# SMOTE
SMOTE = SMOTE(sampling_strategy='minority', random_state=1472)
X, Y = SMOTE.fit_resample(X, Y)
X_nn_train, Y_nn_train = SMOTE.fit_resample(X_nn_train, Y_nn_train)

# Model hyperparameters load
tree = ET.parse(model_dir+'Model/model_parameters.xml')
root = tree.getroot()

loaded_parameters = {}

for model_element in root:
    model_name = model_element.tag
    model_params = {}
    for param_element in model_element:
        param_name = param_element.tag
        param_value = param_element.text
        model_params[param_name] = float(param_value) if param_value.replace('.', '', 1).isdigit() else param_value
    loaded_parameters[model_name] = model_params

LR_load_hpp = loaded_parameters.get('LR_parameters')
SVM_load_hpp = loaded_parameters.get('SVC_parameters')
RF_load_hpp = loaded_parameters.get('RF_parameters')
ET_load_hpp = loaded_parameters.get('ET_parameters')
GBM_load_hpp = loaded_parameters.get('GBM_parameters')
XGB_load_hpp = loaded_parameters.get('XGB_parameters')
NN_load_hpp = loaded_parameters.get('NN_parameters')

# Model training
# LR model
LR_model = LogisticRegression(C=LR_load_hpp.get('C'),
                              random_state=1472)
LR_model.fit(X, Y)
joblib.dump(LR_model, os.path.join(project_models, f'{LR_model.__class__.__name__}.pkl'))

# SVM model
SVC_model = SVC(C=SVM_load_hpp.get('C'),
                gamma=SVM_load_hpp.get('gamma'),
                random_state=1472,
                probability=True)
SVC_model.fit(X, Y)

joblib.dump(SVC_model, os.path.join(project_models, f'{SVC_model.__class__.__name__}.pkl'))

# RF model
RF_model = RandomForestClassifier(max_depth=RF_load_hpp.get('max_depth'),
                                  max_features=RF_load_hpp.get('max_features'),
                                  min_samples_leaf=int(RF_load_hpp.get('min_samples_leaf')),
                                  min_samples_split=int(RF_load_hpp.get('min_samples_split')),
                                  n_estimators=int(RF_load_hpp.get('n_estimators')),
                                  random_state=1472)
RF_model.fit(X, Y)

joblib.dump(RF_model, os.path.join(project_models, f'{RF_model.__class__.__name__}.pkl'))

# ET model
ET_model = ExtraTreesClassifier(max_depth=ET_load_hpp.get('max_depth'),
                                max_features=ET_load_hpp.get('max_features'),
                                min_samples_leaf=int(ET_load_hpp.get('min_samples_leaf')),
                                min_samples_split=int(ET_load_hpp.get('min_samples_split')),
                                n_estimators=int(ET_load_hpp.get('n_estimators')),
                                random_state=1472)
ET_model.fit(X, Y)

joblib.dump(ET_model, os.path.join(project_models, f'{ET_model.__class__.__name__}.pkl'))

# GBM model
GBM_model = GradientBoostingClassifier(learning_rate=GBM_load_hpp.get('learning_rate'),
                                       max_depth=GBM_load_hpp.get('max_depth'),
                                       min_samples_leaf=int(GBM_load_hpp.get('min_samples_leaf')),
                                       min_samples_split=int(GBM_load_hpp.get('min_samples_split')),
                                       n_estimators=int(GBM_load_hpp.get('n_estimators')),
                                       subsample=GBM_load_hpp.get('subsample'),
                                       random_state=1472)
GBM_model.fit(X, Y)

joblib.dump(GBM_model, os.path.join(project_models, f'{GBM_model.__class__.__name__}.pkl'))

# XGBoost model
X_arr = X.values
Y_arr = Y.values

XGB_model = XGBClassifier(colsample_bytree=XGB_load_hpp.get('colsample_bytree'),
                          eta=XGB_load_hpp.get('eta'),
                          gamma=XGB_load_hpp.get('gamma'),
                          max_depth=int(XGB_load_hpp.get('max_depth')),
                          min_child_weight=XGB_load_hpp.get('min_child_weight'),
                          n_estimators=int(XGB_load_hpp.get('n_estimators')),
                          subsample=XGB_load_hpp.get('subsample'),
                          eval_metric=XGB_load_hpp.get('eval_metric'),
                          random_state=1472)
XGB_model.fit(X_arr, Y_arr)

joblib.dump(XGB_model, os.path.join(project_models, f'{XGB_model.__class__.__name__}.pkl'))

# self-attention NN model
input_dim = X_nn_train.shape[1]
initializer = tf.keras.initializers.HeNormal()
regularizer = regularizers.l2(0.01)

epochs = 200
batch_size = int(NN_load_hpp.get('batch_size'))

opt = tf.keras.optimizers.Adam(learning_rate=NN_load_hpp.get('learning_rate'))
auc = AUC()

best_weight_path = os.path.join(project_models, "attention_best.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_weight_path, save_best_only=True, monitor="val_auc", mode='max', save_weights_only=True
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode='max', patience=10, verbose=1)
]

# 모델 세부 설정
inputs = tf.keras.layers.Input(shape=(input_dim,))
dense_v = tf.keras.layers.Dense(input_dim, activation=None)(inputs)
attn_score = tf.keras.layers.Softmax(axis=-1)(dense_v)
cal_score = tf.math.multiply(inputs, attn_score)
Dense1 = tf.keras.layers.Dense(int(NN_load_hpp.get('first_layer')), activation='relu',
                               kernel_initializer=initializer, kernel_regularizer=regularizer)(cal_score)
Dense1_BN = tf.keras.layers.BatchNormalization()(Dense1)
Dense2 = tf.keras.layers.Dense(int(NN_load_hpp.get('second_layer')), activation='relu',
                               kernel_initializer=initializer, kernel_regularizer=regularizer)(Dense1)
Dense2_BN = tf.keras.layers.BatchNormalization()(Dense2)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(Dense2_BN)

NN_model = tf.keras.Model(inputs=inputs, outputs=outputs)

NN_model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=[auc],
)
history = NN_model.fit(
    X_nn_train,
    Y_nn_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_nn_valid, Y_nn_valid),
    callbacks=callbacks
)

NN_model.load_weights(best_weight_path)


# Prediction with test set

df_test = pd.read_csv(f'Data/{test_file}', encoding='cp949')
df_test = df_test[['name', 'smiles']]

test_mols = [Chem.MolFromSmiles(smiles) for smiles in df_test["smiles"]]

none_list = []
for i in range(len(test_mols)):
    if test_mols[i] is None:
        none_list.append(i)
        print('Appended in none_list')

reg_idx = 0
for i in none_list:
    del test_mols[i - reg_idx]
    reg_idx += 1

if len(none_list) != 0:
    df_test = df_test.drop(none_list, axis=0)
    df_test = df_test.reset_index(drop=True)

bit_info_list = []
bit_info = {}
fps = []
b = 0

for a in test_mols:
    fps.append(AllChem.GetMorganFingerprintAsBitVect(a, 3, nBits=2048, bitInfo=bit_info))
    bit_info_list.append(bit_info.copy())
arr_list = []

for i in range(len(fps)):
    array = np.zeros((0,), dtype=np.int8)
    arr_list.append(array)
for i in range(len(fps)):
    bit = fps[i]
    DataStructs.ConvertToNumpyArray(bit, arr_list[i])

test_x = np.stack([i.tolist() for i in arr_list])
test_finprt = pd.DataFrame(test_x)

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

test_input = pd.concat([test_finprt, test_chem], axis=1)

X_test = test_input.iloc[:, :]

X_test_arr = X_test.values

X_test_nn = np.asarray(X_test).astype('float64')

classifiers = {'LR': LR_model,
               'SVM': SVC_model,
               'RF': RF_model,
               'ET': ET_model,
               'GBM': GBM_model,
               'XGB': XGB_model,
               'NN': NN_model}


def model_prediction(classifier_name):
    cls = classifiers.get(classifier_name)
    if cls is XGB_model:
        pred_z = cls.predict_proba(X_test_arr)[:, 1]
    elif cls is NN_model:
        pred_z = cls.predict(X_test_nn)[:, -1]
    else:
        pred_z = cls.predict_proba(X_test)[:, 1]
    output_probability = pd.Series(pred_z[:])

    output_df = pd.concat([df_test[['name', 'smiles']], output_probability], axis=1)
    output_df.columns = ['name', 'smiles', 'probability']
    output_df.to_csv(project_result+f'[{output_name}]_{classifier_name}_predict_result.csv', encoding='cp949',
                     index=False)


for a in classifiers.keys():
    model_prediction(classifier_name=a)

print('-*-*' * 20 + '\nIf you checked this message, the result was generated without any issues.\n')
print(f"The parameters of the trained models were saved in the path '{project_models}'.")
print(f"The CSV file of the prediction results for the test set was saved in the path '{project_result}'.")
print('-*-*' * 20)