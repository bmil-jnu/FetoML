import sys
import os
import IPython
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import *
from sklearn.metrics import *
import keras_tuner
from imblearn.over_sampling import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras import regularizers
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED

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

import joblib
from sklearn.preprocessing import StandardScaler

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
train_chem['HBOND'] = train_qe['HBA']+train_qe['HBD']

# ALOGP, TPSA, HBA+HBD standard scaling
train_chem['ALOGP'] = train_qe['ALOGP']
train_chem['PSA'] = train_qe['PSA']

train_chem[['HBOND', 'ALOGP', 'PSA']] = sds_scaler.fit_transform(train_chem[['HBOND', 'ALOGP', 'PSA']])

scaler_file_name = 'standard_scaler_inform.pkl'
joblib.dump(sds_scaler, scaler_file_name)

# 생성한 데이터 병합
input_df=pd.concat([train_finprt, train_chem, df_train['category']], axis=1)
input_df = input_df.values

# X와 Y 분할
X = input_df[:, : -1]
Y = input_df[:, -1]

# np.asarray astype
X = np.asarray(X).astype('float64')
Y = np.asarray(Y).astype('float64')

# input data의 차원 수
input_dim = X.shape[1]

# Layer weight initializers 설정 (가중치 초기화 설정)
initializer = tf.keras.initializers.HeNormal()

# L2 regularizer 설정
regularizer = regularizers.l2(0.001)


def OneLayerModel(x1):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    dense_v = tf.keras.layers.Dense(input_dim, activation=None)(inputs)
    attn_score = tf.keras.layers.Softmax(axis=-1)(dense_v)
    cal_score = tf.math.multiply(inputs, attn_score)
    Dense1 = tf.keras.layers.Dense(x1, activation='relu',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer)(cal_score)
    Dense1_BN = tf.keras.layers.BatchNormalization()(Dense1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(Dense1_BN)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def TwoLayerModel(x1, x2):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    dense_v = tf.keras.layers.Dense(input_dim, activation=None)(inputs)
    attn_score = tf.keras.layers.Softmax(axis=-1)(dense_v)
    cal_score = tf.math.multiply(inputs, attn_score)
    Dense1 = tf.keras.layers.Dense(x1, activation='relu',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer)(cal_score)
    Dense1_BN = tf.keras.layers.BatchNormalization()(Dense1)
    Dense2 = tf.keras.layers.Dense(x2, activation='relu',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer)(Dense1_BN)
    Dense2_BN = tf.keras.layers.BatchNormalization()(Dense2)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(Dense2_BN)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def ThreeLayerModel(x1, x2, x3):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    dense_v = tf.keras.layers.Dense(input_dim, activation=None)(inputs)
    attn_score = tf.keras.layers.Softmax(axis=-1)(dense_v)
    cal_score = tf.math.multiply(inputs, attn_score)
    Dense1 = tf.keras.layers.Dense(x1, activation='relu',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer)(cal_score)
    Dense1_BN = tf.keras.layers.BatchNormalization()(Dense1)
    Dense2 = tf.keras.layers.Dense(x2, activation='relu',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer)(Dense1_BN)
    Dense2_BN = tf.keras.layers.BatchNormalization()(Dense2)
    Dense3 = tf.keras.layers.Dense(x3, activation='relu',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer)(Dense2_BN)
    Dense3_BN = tf.keras.layers.BatchNormalization()(Dense3)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(Dense3_BN)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


hp = keras_tuner.HyperParameters()

def model_builder(hp):
    auc = AUC()

    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)
    layer1_units = hp.Int('dense_1', min_value=32, max_value=128, step=4)
    layer_last_units = hp.Int('dense_2', min_value=4, max_value=32, step=4)

    if num_dense_layers == 1:
        model = OneLayerModel(layer1_units)
    elif num_dense_layers == 2:
        model = TwoLayerModel(layer1_units, layer_last_units)
    elif num_dense_layers == 3:
        model = ThreeLayerModel(layer1_units, layer_last_units, layer_last_units)
    else:
        raise

    hp_learning_rate = hp.Float('learning_rate', min_value=1e-3, max_value=1e-2, step=1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=[auc])

    return model


class CVTuner(keras_tuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, batch_size=32, epochs=100):
        val_auc = []
        for train, test in skf.split(X, Y):
            X_train = X[train]
            Y_train = Y[train]
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
            X_test = X[test]
            Y_test = Y[test]
            model = self.hypermodel.build(trial.hyperparameters)
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    "best_model_weight.h5", save_best_only=True,
                    monitor='val_auc',
                    mode='max',
                    save_weights_only=True),
                tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', verbose=1)]
            model.fit(X_train,
                      Y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(X_test, Y_test),
                      callbacks=callbacks)
            print('load_weights')
            model.load_weights("best_model_weight.h5")
            print('model evaluation')
            preds = model.predict(X_test)
            val_auc.append(roc_auc_score(Y_test, preds))
            print('remove_weight_file')
            os.remove(f"best_model_weight.h5")
        self.oracle.update_trial(trial.trial_id, {'val_auc': np.mean(val_auc)})
        self.save_model(trial.trial_id, model)


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


n_fold = 5
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1472)
smote = SMOTE(sampling_strategy='minority', random_state=1472)  # SMOTE

tuner = CVTuner(
    hypermodel=model_builder,
    oracle=keras_tuner.oracles.BayesianOptimization(
        objective=keras_tuner.Objective('val_auc', direction='max'),
        max_trials=200),
    project_name='bo_attention_nn')

tuner.search(X, Y, batch_size=32, epochs=100)

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
print("Best Hyperparameters")
print(best_hyperparameters.values)