import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import *
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from sklearn.metrics import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import AUC

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

# 훈련용 데이터 셋으로 입력 가능하도록 변환
input_df = input_df.values

# X와 Y 분할
X = input_df[:, : -1]
Y = input_df[:, -1]

# np.asarray astype
X = np.asarray(X).astype('float64')
Y = np.asarray(Y).astype('float64')

# 10% hold-out
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.12, random_state=1472)

Y_test_count = pd.DataFrame(Y_test)
print(Y_test_count.value_counts())

# SMOTE
SMOTE = SMOTE(sampling_strategy='minority', random_state=1472)
X_train, Y_train = SMOTE.fit_resample(X_train, Y_train)

print(pd.DataFrame(Y_train).value_counts())

# 가중치 조절
class_weight = {1 : 1, 0 : 1}

# input data의 차원 수
input_dim = X_train.shape[1]

# Layer weight initializers 설정 (가중치 초기화 설정)
initializer = tf.keras.initializers.HeNormal()

# L2 regularizer 설정
regularizer = regularizers.l2(0.001)

# 모델 하이퍼 파라미터 (epochs, batch_size)
epochs = 200
batch_size = 32

opt = tf.keras.optimizers.Adam(learning_rate=0.003)
auc = AUC()

# callbacks 옵션
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "attention_best.h5", save_best_only=True, monitor="val_auc", mode='max', save_weights_only = True
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode='max', patience=10, verbose=1)
]

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

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=[auc],
)

history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, Y_test),
    callbacks=callbacks,
    class_weight=class_weight
)

# 최적 모델 불러온 후 성능 확인
model.load_weights("attention_best.h5")
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

# 모델 학습과정 그래프 그리기
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = ' ', c= 'red', label = 'testset_loss')
plt.plot(x_len, y_loss, marker = ' ', c = 'blue', label = 'trainset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# ROC curve
preds = model.predict(X_test)

fpr, tpr, threshold = roc_curve(Y_test, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# precision, recall, threshold 생성
precision, recall, thresholds = precision_recall_curve(Y_test, preds)
# pr_auc 정의
pr_auc = auc(recall, precision)
# F1 score 정의
F1 = 2*(precision*recall)/(precision+recall)

plt.title('PR curve')
plt.plot(recall, precision, 'b', label = 'AUC = %0.2f' % pr_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# best threshold를 탐색하기 위한 PR-curve 표 만들기
th_precision = pd.DataFrame(precision, columns = ['precision'])
th_recall = pd.DataFrame(recall, columns = ['recall'])
th_threshold = pd.DataFrame(thresholds, columns = ['threshold'])
th_F1 = pd.DataFrame(F1, columns = ['F1 Score'])
th = pd.concat([th_precision, th_recall, th_threshold,th_F1], axis = 1)

bestloc = th['F1 Score'].argmax()

threshold = th['threshold'].loc[bestloc]
pred_z = model.predict(X_test)
print(len(pred_z[pred_z >= threshold]))
print(len(pred_z[pred_z < threshold]))
pred_z[pred_z >= threshold] = 1
pred_z[pred_z < threshold] = 0
pred_roc = model.predict(X_test)

Precision = precision_score(Y_test,pred_z)
Recall = recall_score(Y_test,pred_z)

print('Best threshold : {0}'.format(th['threshold'].loc[bestloc]))
print('accuracy : {0}'.format(accuracy_score(Y_test,pred_z)))
print('Precision : {0}'.format(Precision))
print('Recall : {0}'.format(Recall))
print('ROC_score : {0}'.format(roc_auc_score(Y_test,pred_roc)))

F1_score = 2*(Precision*Recall)/(Precision+Recall)
print("F1 score : {0}".format(F1_score))