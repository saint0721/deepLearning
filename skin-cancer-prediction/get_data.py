import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(base_dir, imageid_path_dict):
    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    df_original[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    df_original['duplicates'] = df_original['lesion_id']
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df_original['train_or_val'] = df_original['image_id']
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)

    df_train = df_original[df_original['train_or_val'] == 'train']

    data_aug_rate = [15,10,5,50,0,40,5]
    for i in range(7):
        if data_aug_rate[i]:
            pd.concat(
        [df_train] + [df_train.loc[df_train['cell_type_idx'] == i,:]] * (data_aug_rate[i] - 1), ignore_index=True)

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    return df_train, df_val