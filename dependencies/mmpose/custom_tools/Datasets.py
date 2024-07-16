import glob
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
import os.path as osp
import datetime




def combin_pkl(path, pklname='result_train.pkl'):
    result = []
    path = glob.glob(path + "/*.pkl")
    for d in tqdm(path):
        if d.endswith('.pkl'):
            with open(d, 'rb') as f:
                content = pickle.load(f)
            result.append(content)
    with open(pklname, 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)

name_cls = 'Zebra'
# files_path1 = f'extract_split/{name_cls}/pkl_output_{name_cls}A800_train'
# files_path2 = f'extract_split/{name_cls}/pkl_output_{name_cls}A800_val'
# files_path3 = f'extract_split/{name_cls}/pkl_output_{name_cls}A800_test'
time_format = datetime.datetime.now().strftime('%Y%m%d')


# files_list = glob.glob(files_path + "/*.pkl")
# print(files_list)
# train_path, val_path = train_test_split(files_list, test_size=0.2)
# train_path = glob.glob(files_path1 + "/*.pkl")
# val_path = glob.glob(files_path2 + "/*.pkl")
# test_path = glob.glob(files_path3 + "/*.pkl")
root = 'extract_combination'

if not osp.exists(f'{root}/{name_cls}/'):
    os.makedirs(f'{root}/{name_cls}/')
# pkl_cls2_Black_Bear_Autodataset_A800_test
train_path = '/home/ztx/lj/Animalbehavior/mmpose/custom_tools/extract_split2/pkl_4leg_Zebra_Autodataset_A800_train'
val_path = '/home/ztx/lj/Animalbehavior/mmpose/custom_tools/extract_split2/pkl_4leg_Zebra_Autodataset_A800_val'
test_path = '/home/ztx/lj/Animalbehavior/mmpose/custom_tools/extract_split2/pkl_4leg_Zebra_Autodataset_A800_test'

combin_pkl(train_path, f'{root}/{name_cls}/{name_cls}_bigdatasets_result_a800train{time_format}.pkl')
combin_pkl(val_path, f'{root}/{name_cls}/{name_cls}_bigdatasets_result_a800val{time_format}.pkl')
combin_pkl(test_path, f'{root}/{name_cls}/{name_cls}_bigdatasets_result_a800test{time_format}.pkl')




# name_list = ['30', '60', '90', '120', '150', '180', '210',
#              '240', '268']
#
# for nl in name_list:
#     new_train_path = f'extract_split/{name_cls}/pkl_output_{name_cls}A800_train_{nl}'
#     train_path = glob.glob(new_train_path + "/*.pkl")
#     if not osp.exists(root):
#         os.makedirs(root)
#     combin_pkl(train_path, f'{root}/{name_cls}/{name_cls}_{nl}_bigdatasets_result_a800train{time_format}.pkl')