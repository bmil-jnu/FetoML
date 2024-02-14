import os
import sys
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

warnings.filterwarnings(action='ignore')
sys.path.append('C:/FetoML/Model Code')

df_train = pd.read_csv("../../Data/fetal_toxicity_Train.csv", encoding = 'cp949')
df_train = df_train[['name','smiles','category']]

df_test = pd.read_csv("../../Data/fetal_toxicity_Test.csv", encoding = 'cp949')
df_test = df_test[['name','smiles','category']]

df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# mol 형태로 변환
mols = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]

# mol 형태로 변환이 되지 않은 경우, none_list에 담는다
none_list = []
for i in range(len(mols)):
    if mols[i] is None:
        none_list.append(i)
        print('none_list에 추가됨')

reg_idx = 0
for i in none_list:
    del mols[i - reg_idx]
    reg_idx += 1

# none_list가 존재할 경우, 삭제 후 데이터프레임 인덱스 맞춰주기
if len(none_list) != 0:
    df = df.drop(none_list, axis=0)
    df = df.reset_index(drop=True)

# fingerprint 생성
bit_info_list = []  # bit vector의 설명자 리스트 담기
bit_info = {}  # bit vector 설명자
fps = []

b = 0
# mol 파일에서 fingerprint Bit Vector 형태로 변환하기
for a in mols:
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

mols_x = np.stack([i.tolist() for i in arr_list])
input_df = pd.DataFrame(mols_x)

def DrawFingerprint(mols=None, dataframe=None, bit_info_list=None, bit_index=None):
    # 디렉토리 명
    parent_dir_name = 'Feature_importance_analysis_result'
    sub_dir_name = f'/bit_{bit_index}'
    dir_name_tosave = parent_dir_name + sub_dir_name

    # 저장할 디렉토리 생성
    if os.path.isdir(parent_dir_name) == False:
        os.mkdir(parent_dir_name)
    else:
        if os.path.isdir(dir_name_tosave) == False:
            os.mkdir(dir_name_tosave)
        else:
            pass

    # Check the Input Values
    if mols is None:
        raise ValueError('Please input the \'mols\' value, it must be list of mol.')

    if dataframe is None:
        raise ValueError('Please input the \'dataframe\' value, it must be dataframe.')

    if bit_info_list is None:
        raise ValueError('Please input the \'bit_info_list\' value, it must be list of bit_info.')

    if bit_index is None:
        raise ValueError('Please input the \'bit_index\' value, it must be integer')

    # bit_index==1에 해당하는 분자들 가져오기
    selected_drug = dataframe[dataframe[bit_index] == 1]
    selected_drug_list = list(selected_drug.index)

    # 해당 drug의 bit_index에 대한 substructure 가져오기
    SMILES_inform = {}  # SMILES 담을 곳
    for idx in selected_drug_list:
        selected_env = Chem.FindAtomEnvironmentOfRadiusN(mols[idx], bit_info_list[idx][bit_index][0][1],
                                                         bit_info_list[idx][bit_index][0][0])
        selected_submol = Chem.PathToSubmol(mols[idx], selected_env)
        selected_SMILES = Chem.MolToSmiles(selected_submol)
        if selected_SMILES is '':
            pass
        else:
            if SMILES_inform.get(selected_SMILES) is None:
                SMILES_inform[selected_SMILES] = idx
            else:
                pass

    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    drawOptions.legendFraction = 0.1

    for key, value in SMILES_inform.items():
        try:
            draw_fingerprint = Draw.DrawMorganBit(mols[value],
                                                  bit_index,
                                                  bit_info_list[value],
                                                  #                                               legend=f'SMILES: {key}',
                                                  molSize=(600, 600),
                                                  useSVG=False,
                                                  drawOptions=drawOptions)
            display(draw_fingerprint)
            print(key)
            draw_fingerprint.save(dir_name_tosave + f'/{key}.png')
        except:
            print(f'{key}, error ocurred')
            pass

DrawFingerprint(mols=mols,
                dataframe=input_df,
                bit_info_list=bit_info_list,
                bit_index=1)

DrawFingerprint(mols=mols,
                dataframe=input_df,
                bit_info_list=bit_info_list,
                bit_index=80)