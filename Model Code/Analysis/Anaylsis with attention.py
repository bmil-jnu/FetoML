import sys
import cairosvg
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from sklearn.metrics import *
from tensorflow.keras import regularizers

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
test_input_df=pd.concat([test_finprt, test_chem, df_test['category']], axis=1)

# 훈련용 데이터 셋으로 입력 가능하도록 변환
test_input=test_input_df.values

# X와 Y 분할
X_ex = test_input[:, :-1]
Y_ex = test_input[:, -1]

# np.asarray astype
X_ex = np.asarray(X_ex).astype('float64')
Y_ex = np.asarray(Y_ex).astype('float64')

# input data의 차원 수
input_dim = X_ex.shape[1]

# Layer weight initializers 설정 (가중치 초기화 설정)
initializer = tf.keras.initializers.HeNormal()

# L2 regularizer 설정
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

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.load_weights("../Model/attention_best.h5")

preds = model.predict(X_ex)

precision, recall, thresholds = precision_recall_curve(Y_ex, preds)
F1 = 2*(precision*recall)/(precision+recall)
th_precision = pd.DataFrame(precision, columns = ['precision'])
th_recall = pd.DataFrame(recall, columns = ['recall'])
th_threshold = pd.DataFrame(thresholds, columns = ['threshold'])
th_F1 = pd.DataFrame(F1, columns = ['F1 Score'])
th = pd.concat([th_precision, th_recall, th_threshold, th_F1], axis = 1)

bestloc = th['F1 Score'].argmax()
best_threshold = threshold = th['threshold'].loc[bestloc]

print(f'Best threshold: {best_threshold}')

test_preds = preds.copy()

test_preds[test_preds >= best_threshold] = 1
test_preds[test_preds < best_threshold] = 0

Precision = precision_score(Y_ex,test_preds)
Recall = recall_score(Y_ex,test_preds)

print('accuracy : {0}'.format(accuracy_score(Y_ex ,test_preds)))
print('Precision : {0}'.format(Precision))
print('Recall : {0}'.format(Recall))
print('ROC_score : {0}'.format(roc_auc_score(Y_ex, preds)))

F1_score = 2*(Precision*Recall)/(Precision+Recall)
print("F1 score : {0}".format(F1_score))

def PredRank(dataframe, data, model):
    pred = model.predict(data)
    df_pred = pd.DataFrame(pred, columns = ['score'])
    df_pred = pd.concat([df_pred, dataframe[['category', 'name']]], axis=1)
    pred_rank = df_pred.sort_values('score', ascending=False)
    return pred_rank


def MakeAttentionScore(dataframe, data, model, fp_size, thresholds=0.5, label="true_pos"):
    global attention_score
    attn_ = tf.keras.Model(inputs=model.inputs, outputs=model.layers[2].output)  # attention layer

    attention_score = attn_.predict(data)
    attention_score[:, :fp_size] = attention_score[:, :fp_size] * data[:, :fp_size]
    attention_score = pd.DataFrame(attention_score)

    if label == "true_pos":
        pred = model.predict(data)
        pred[pred >= thresholds] = 1
        pred[pred < thresholds] = 0
        pred = pd.DataFrame(pred)
        pred_idx = pred[pred[0] == 1].index
        dataframe = dataframe.loc[pred_idx]
        true_positive_idx = dataframe[dataframe['category'] == 1].index
        attention_score = attention_score.loc[true_positive_idx]
        attention_score = attention_score.transpose()

    elif label == "pred_pos":
        pred = model.predict(data)
        pred[pred >= thresholds] = 1
        pred[pred < thresholds] = 0
        pred = pd.DataFrame(pred)
        pred_idx = pred[pred[0] == 1].index

        attention_score = attention_score.loc[pred_idx]
        attention_score = attention_score.transpose()

    elif label == "all":
        attention_score = attention_score.transpose()

    return attention_score

def MakeAttentionIndex(attention_score, dataframe):
    attention_index = attention_score.transpose().index
    inform = dataframe.loc[attention_index]
    return inform

def FindDrug(inform, drug_name):
    if type(drug_name) == str:
        selected_drug = inform[inform['name']==drug_name]
        print('INDEX')
        print(selected_drug)
    else:
        raise ValueError('The \'drug_name\' value must be string type')


def AttentionAnalysis(mols=None, attention_score=None, dataframe=None, bit_info_list=None, index=None, drug_name=None,
                      rank=10, fp_size=None):
    highlightAtomLists = []  # 하위 분자 구조 하이라이트 정보 리스트에 담기
    rank_information = []
    legends = []  # 하위 분자 구조 정보 담을 곳

    # Check the Input Values
    if mols is None:
        raise ValueError('Please input the \'mols\' value, it must be list of mol.')

    if attention_score is None:
        raise ValueError('Please input the \'attention_score\' value, it must be dataframe.')

    if dataframe is None:
        raise ValueError('Please input the \'dataframe\' value, it must be dataframe.')

    if bit_info_list is None:
        raise ValueError('Please input the \'bit_info_list\' value, it must be list of bit_info.')

    if rank <= 0:
        raise ValueError('The \'rank\' value must be larger than 0, please start with 1.')

    if fp_size is None:
        raise ValueError(
            'Please input the \'fp_size\' value, it must be integer.\nThe vector size is structural fingerprint size')

    if (index is None) and (drug_name is None):
        raise (
            'Both values \'index\' and \'drug_name\' are None, please input the only one of the index value or drug_name value.')
    elif (index is not None) and (drug_name is not None):
        raise (
            'Both values \'index\' and \'drug_name\' were entered, please input the only one of the index value or drug_name value.')

    # Molecule Drawing with Highlighted Hit Sub-structure
    if index != None:
        if type(index) == int:
            selected_score = attention_score.loc[:, index]
            selected_score = selected_score.sort_values(ascending=False)
            for rank_i in range(rank):
                if selected_score.index[rank_i] < fp_size:  # index가 fp_size보다 작으면 molecular의 하위분자 구조, 분자 그리기
                    selected_bit = selected_score.index[rank_i]
                    if bit_info_list[index][selected_bit][0][
                        1] == 0:  # Radius가 0이면 0이 아닌것 찾을 때까지, 만약 못찾으면 그냥 0인 상태로 하위분자 그리기
                        check_bits = 0
                        while bit_info_list[index][selected_bit][check_bits][1] == 0:
                            check_bits += 1
                            if check_bits == len(bit_info_list[index][selected_bit]):
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits - 1][1],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits - 1][0])
                                break
                            elif bit_info_list[index][selected_bit][check_bits][1] != 0:
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits][1],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits][0])
                                break
                    else:
                        selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index],
                                                                         bit_info_list[index][selected_bit][0][1],
                                                                         bit_info_list[index][selected_bit][0][0])
                    selected_submol = Chem.PathToSubmol(mols[index], selected_env)
                    highlightAtom = mols[index].GetSubstructMatch(selected_submol)
                    highlightAtomLists.append(highlightAtom)
                    selected_SMILES = Chem.MolToSmiles(selected_submol)
                    selected_legend = f'Rank: {rank_i + 1}, bit: {selected_bit}'
                    rank_information.append(
                        f'Rank: {rank_i + 1}\nAttention score: {selected_score.iloc[rank_i]:.5f}\nbit: {selected_bit}\nbit_info: {bit_info_list[index][selected_bit]}\nSMILES: {selected_SMILES}')
                    legends.append(str(selected_legend))
                elif selected_score.index[rank_i] >= fp_size:  # index가 fp_size보다 크거나 같으면 chemical feature
                    selected_bit = selected_score.index[rank_i]
                    selected_legend = f'Rank: {rank_i + 1}, bit: {selected_bit}'
                    rank_information.append(
                        f'Rank: {rank_i + 1}\nAttention score: {selected_score.iloc[rank_i]:.5f} \nbit: {selected_bit}')
                    legends.append(str(selected_legend))
                    highlightAtomLists.append('')
            Draw_attention_score = Chem.Draw.MolsToGridImage([mols[index] for i in range(len(highlightAtomLists))],
                                                             molsPerRow=2,
                                                             subImgSize=(400, 400),
                                                             highlightAtomLists=highlightAtomLists,
                                                             legends=legends,
                                                             useSVG=True)
            for i in range(len(rank_information)):
                print(rank_information[i])
            return Draw_attention_score
        elif type(index) != int:
            raise ValueError('The \'index\' value must be integer type')

    elif drug_name != None:
        if type(drug_name) == str:
            selected_drug = dataframe[dataframe['name'] == drug_name]
            index = selected_drug.index
            selected_score = attention_score.loc[:, index]
            selected_score = selected_score.sort_values(ascending=False)
            for rank_i in range(rank):
                if selected_score.index[rank_i] < fp_size:  # index가 fp_size보다 작으면 molecular의 하위분자 구조, 분자 그리기
                    selected_bit = selected_score.index[rank_i]
                    if bit_info_list[index][selected_bit][0][
                        1] == 0:  # Radius가 0이면 0이 아닌것 찾을 때까지, 만약 못찾으면 그냥 0인 상태로 하위분자 그리기
                        check_bits = 0
                        while bit_info_list[index][selected_bit][check_bits][1] == 0:
                            check_bits += 1
                            if check_bits == len(bit_info_list[index][selected_bit]):
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits - 1][1],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits - 1][0])
                                break
                            elif bit_info_list[index][selected_bit][check_bits][1] != 0:
                                selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits][1],
                                                                                 bit_info_list[index][selected_bit][
                                                                                     check_bits][0])
                                break
                    else:
                        selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[index],
                                                                         bit_info_list[index][selected_bit][0][1],
                                                                         bit_info_list[index][selected_bit][0][0])
                    selected_submol = Chem.PathToSubmol(mols[index], selected_env)
                    highlightAtom = mols[index].GetSubstructMatch(selected_submol)
                    highlightAtomLists.append(highlightAtom)
                    selected_SMILES = Chem.MolToSmiles(selected_submol)
                    selected_legend = f'Rank: {rank_i + 1}, bit: {selected_bit}'
                    rank_information.append(
                        f'Rank: {rank_i + 1}\nAttention score: {selected_score.iloc[rank_i]:.5f}\nbit: {selected_bit}\nbit_info: {bit_info_list[index][selected_bit]}\nSMILES: {selected_SMILES}')
                    legends.append(str(selected_legend))
                elif selected_score.index[rank_i] >= fp_size:  # index가 fp_size보다 크거나 같으면 chemical feature
                    selected_bit = selected_score.index[rank_i]
                    selected_legend = f'Rank: {rank_i + 1}, bit: {selected_bit}'
                    rank_information.append(
                        f'Rank: {rank_i + 1}\nAttention score: {selected_score.iloc[rank_i]:.5f} \nbit: {selected_bit}')
                    legends.append(str(selected_legend))
                    highlightAtomLists.append('')
            Draw_attention_score = Chem.Draw.MolsToGridImage([mols[index] for i in range(len(highlightAtomLists))],
                                                             molsPerRow=2,
                                                             subImgSize=(400, 400),
                                                             highlightAtomLists=highlightAtomLists,
                                                             legends=legends,
                                                             useSVG=True)
            for i in range(len(rank_information)):
                print(rank_information[i])
            return Draw_attention_score
        elif type(drug_name) != str:
            raise ValueError('The \'drug_name\' value must be string type')

pd.set_option('display.max_rows', None)
pred_rank = PredRank(df_test, X_ex, model)
pred_rank.sort_values(by='score', ascending=False)

attention_score = MakeAttentionScore(dataframe=df_test,
                                     data=X_ex,
                                     model=model,
                                     fp_size=2048,
                                     thresholds=0.5,
                                     label="all")

for i in range(30):
    idx = int(pred_rank.iloc[i].name)
    drug_name = pred_rank["name"].iloc[i]

    print(f'\n\n\n{drug_name}\n')

    draw = AttentionAnalysis(mols=test_mols,
                             attention_score=attention_score,
                             dataframe=df_test,
                             bit_info_list=bit_info_list,
                             index=idx,
                             rank=15,
                             fp_size=2048)

    cairosvg.svg2pdf(bytestring=draw.data, write_to=f"Attention analysis result/Rank {i + 1}_{drug_name}.pdf")