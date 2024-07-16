import re
import json
from pathlib import Path as p


def mkgdjson(videopath, name_cls=None):
    videopath = p(videopath)
    act_categories = []
    act_idx = dict()
    categories_id = 0
    for i in videopath.iterdir():
        act_categories.append(i.stem.title())
        act_idx[i.stem] = categories_id
        categories_id = categories_id + 1

    Info =dict(
        categories=act_categories
        # annotations=file_info
    )
    # print(Info)
    json_path = f'./{name_cls}_behaviorcategory_annotation.json'
    with open(json_path, 'w') as f:
        json.dump(Info, f)

    return Info


def loadgroundtruth_json(jsonpath):
    jsonpath = f'./{jsonpath}_behaviorcategory_annotation.json'
    with open(jsonpath, 'r') as f:
        j_file = json.load(f)

    return j_file




if __name__ == '__main__':
    name_cls = 'primate'
    videopath = r'path/to/video_directory'
    mkgdjson(videopath, name_cls)

