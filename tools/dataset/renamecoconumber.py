from pathlib import Path
from tqdm import tqdm
import json


def json_file_rename(dir_path):
    '''
    According to coco dataset format,
    counting from 0, change the file name of the image and json
    to the same numeric name, and change the ''imgPath''
    in the json according to the new image name.
    '''
    jsons = list(dir_path.rglob('**/*.json'))
    bar = tqdm(total=int(len(jsons)))
    num = 0
    for i in jsons:
        if (Path(i.parent) / f'{i.stem}.jpg').exists() and i.exists():
            new_name_num = f'{num:012d}'
            num += 1
            i.rename(i.parent / f'{new_name_num}.json')
            with open(i.parent / f'{new_name_num}.json', 'r') as f:
                data = json.load(f)
            data['imagePath'] = f'{new_name_num}.jpg'
            with open(i.parent / f'{new_name_num}.json', 'w') as f:
                json.dump(data, f)
            (i.parent / f'{i.stem}.jpg').rename(i.parent / f'{new_name_num}.jpg')
            bar.update()


if __name__ == '__main__':
    # json files directory
    dir_path = 'path/to/your/json_file_dir'
    dir_path = Path(dir_path)
    json_file_rename(dir_path)
