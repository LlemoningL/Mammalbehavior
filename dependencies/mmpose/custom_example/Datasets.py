import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
import os.path as osp
import datetime




def combin_pkl(path, pklname='result_train.pkl'):
    result = []
    for d in tqdm(path):
        if d.endswith('.pkl'):
            with open(d, 'rb') as f:
                content = pickle.load(f)
            result.append(content)
    with open(pklname, 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)

name_cls = 'gsm'
files_path1 = f'extract_split/{name_cls}/pkl_output_{name_cls}A800_train'

time_format = datetime.datetime.now().strftime('%Y%m%d')



root = 'extract_combination'



train_path = 'path/to/trainset'
val_path = 'path/to/valset'
test_path = 'path/to/testset'

combin_pkl(train_path, f'{root}/{name_cls}/{name_cls}_train{time_format}.pkl')
combin_pkl(val_path, f'{root}/{name_cls}/{name_cls}_val{time_format}.pkl')
combin_pkl(test_path, f'{root}/{name_cls}/{name_cls}_test{time_format}.pkl')




