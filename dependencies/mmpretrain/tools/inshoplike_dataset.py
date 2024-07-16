from pathlib import Path as p
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

def inshoplike_dataset(input='', output='inshoplike_dataset'):

    input = p(input)
    data = list(p(input).rglob('*.*'))
    for i in data:
        if i.suffix not in ['.jpg', ' .jpeg', '.png', '.bmp', '.webp',
                            '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP']:
            data.remove(str(i))
    inshop_name = f'{input.stem}_{output}'
    eval_dir = p(input.parent) / inshop_name / 'Eval'
    img_dir = p(input.parent) / inshop_name / 'Img'
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True)
    if not img_dir.exists():
        img_dir.mkdir(parents=True)
    txt_eval = eval_dir / 'list_eval_partition.txt'
    with open(txt_eval, 'w') as f:
        f.write(f'{len(data)}\nimage_name item_id evaluation_status\n')
    with open(txt_eval, 'a') as f:
        for i in input.iterdir():
            if i.stem == 'train':
                for j in tqdm(list(i.iterdir()), postfix=i.stem):
                    if not p(img_dir / j.stem).exists():
                        p(img_dir / j.stem).mkdir(parents=True)
                    for k in j.iterdir():
                        img_new_name = img_dir / j.stem / k.name
                        shutil.copy(str(k), str(img_new_name))
                        line = f'{img_dir.stem / img_new_name} {j.stem} train\n'
                        f.write(line)
            elif i.stem == 'val':
                for j in tqdm(list(i.iterdir()), postfix=i.stem):
                    if not p(img_dir / j.stem).exists():
                        p(img_dir / j.stem).mkdir(parents=True)
                    for k in j.iterdir():
                        img_new_name = img_dir / j.stem / k.name
                        shutil.copy(str(k), str(img_new_name))
                        line = f'{img_dir.stem / img_new_name} {j.stem} query\n'
                        f.write(line)
            elif i.stem == 'test':
                 for j in tqdm(list(i.iterdir()), postfix=i.stem):
                    if not p(img_dir / j.stem).exists():
                        p(img_dir / j.stem).mkdir(parents=True)
                    for k in j.iterdir():
                        img_new_name = img_dir / j.stem / k.name
                        shutil.copy(str(k), str(img_new_name))
                        line = f'{img_dir.stem / img_new_name} {j.stem} gallery\n'
                        f.write(line)


def inshoplike_dataset_in_all(input, train_size=0.7, val_size=0.1, test_size=0.2, output='inshoplike_dataset',
                              seed: int = 42):

    # data = (p(input).rglob('*.{jpg,jpeg,png,bmp,webp,JPG,JPEG,PNG,BMP,WEBP}'))    
    input = p(input)
    inshop_name = f'{input.stem}_{output}'
    eval_dir = p(input.parent) / inshop_name / 'Eval'
    img_dir = p(input.parent) / inshop_name / 'Img'
    print(f'copy images to {img_dir}, please wait')
    shutil.copytree(input, img_dir)
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True)
    if not img_dir.exists():
        img_dir.mkdir(parents=True)

    train_path = list()
    val_path = list()
    test_path = list()
    for dir in img_dir.iterdir():
        data = list(dir.rglob('*.*'))
        # data = p(input).rglob('*')
        for i in data:
            # print(i)
            if i.suffix not in ['.jpg', ' .jpeg', '.png', '.bmp', '.webp',
                                '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP']:
                data.remove(str(i))
        train_p, val_p, test_p = split_dataset(data, train_size, val_size, test_size, seed)
        train_path.extend(train_p)
        val_path.extend(val_p)
        test_path.extend(test_p)
    

    txt_eval = eval_dir / 'list_eval_partition.txt'
    with open(txt_eval, 'w') as f:
        f.write(f'{len(train_path)+len(val_path)+len(test_path)}\nimage_name item_id evaluation_status\n')
    with open(txt_eval, 'a') as f:
        print(f'write train set info to {txt_eval}')
        for tr in train_path:
            img_new_name = img_dir / tr.parent / tr.name
            line = f'{img_new_name} {tr.parent.stem} train\n'
            f.write(line)
        print(f'write query(val) set info to {txt_eval}')
        for va in val_path:
            img_new_name = img_dir / va.parent / va.name
            line = f'{img_new_name} {va.parent.stem} query\n'
            f.write(line)
        print(f'write gallery(test) set info to {txt_eval}')
        for te in test_path:
            img_new_name = img_dir / te.parent / te.name
            line = f'{img_new_name} {te.parent.stem} gallery\n'
            f.write(line)


def split_dataset(data, train_size, val_size, test_size, seed):
    total_num = len(data)
    if train_size == 0:
        train_path = []
        # train_num = round(train_size * total_num)
        val_num = round(val_size * total_num)
        test_num = round(test_size * total_num)
        test_num = min(test_num, total_num - val_num)
        val_path, test_path = train_test_split(data, train_size=val_num, test_size=test_numra, random_state=seed)
    elif val_size == 0:
        val_path = []
        train_num = round(train_size * total_num)
        # val_num = round(val_size * total_num)
        test_num = round(test_size * total_num)
        test_num = min(test_num, total_num - train_num)
        train_path, test_path = train_test_split(data, train_size=train_num, test_size=test_num, random_state=seed)
    elif test_size == 0:
        test_path = []
        train_num = round(train_size * total_num)
        val_num = round(val_size * total_num)
        # test_num = round(test_size * total_num)
        val_num = min(val_num, total_num - train_num)
        train_path, val_path = train_test_split(data, train_size=train_num, test_size=val_num, random_state=seed)
    else:
        train_num = round(train_size * total_num)
        val_num = round(val_size * total_num)
        test_num = round(test_size * total_num)
        test_num = min(test_num, total_num - train_num - val_num)
        train_path, val_path1 = train_test_split(data, train_size=train_num, test_size=total_num - train_num,
                                                 random_state=seed)
        val_path, test_path = train_test_split(val_path1, train_size=val_num, test_size=test_num, random_state=seed)

    return train_path, val_path, test_path


def remove_spaces(path):
    # 检查文件夹是否存在
    if not os.path.exists(path):
        print(f"文件夹 {path} 不存在")
        return

        # 遍历文件夹中的文件和子文件夹
    for root, dirs, files in os.walk(path):
        # 遍历文件夹中的文件
        for file in files:
            # 检查文件名是否包含空格
            if " " in file:
                # 构建新文件名，去掉空格
                new_file = file.replace(" ", "")
                # 重命名文件
                shutil.move(os.path.join(root, file), os.path.join(root, new_file))
                print(f"已将 {file} 重命名为 {new_file}")

            # 指定要检查的文件夹路径

    print('Done, nothing wrong.')

# folder_path = r"/home/gst/14tdisk/14-4tdisk/QLZ_ft_data"  # 请替换为您的文件夹路径
# remove_spaces(folder_path)


# input_path = r'/home/gst/lj/facetree/data/2023年暑期数据采集汇总/facetree/facedataset/priamtes_faceimgs'
input_path = r'/home/gst/lj/facetree/data/2023年暑期数据采集汇总/facetree/facedataset/priamtes_faceimgs'
# # inshoplike_dataset(input_path )
inshoplike_dataset_in_all(input_path, output='inshop_like_dataset')



