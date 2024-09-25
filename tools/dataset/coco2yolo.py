from ultralytics.data.converter import convert_coco



label_dir = 'path/to/your/coco_dataset/annotations/'
output_dir = 'path/to/save'

convert_coco(labels_dir=label_dir,
             save_dir=output_dir,
             use_segments=False,
             use_keypoints=False,
             cls91to80=False)
