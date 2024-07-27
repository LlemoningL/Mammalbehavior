from argparse import ArgumentParser
from utils.VideoProcessor import VideoProcessor
from yacs.config import CfgNode as CN

def default():
    _C = CN()
    _C.PRE = False
    _C.MODEL = CN()

    _C.MODEL.FACE = CN()
    _C.MODEL.FACE.weight = ''
    _C.MODEL.FACE.trt_engine = ''

    _C.MODEL.BODY = CN()
    _C.MODEL.BODY.weight = ''
    _C.MODEL.BODY.trt_engine = ''
    _C.MODEL.BODY.reid_encoder = ''
    _C.MODEL.BODY.r_e_trt_engine = ''
    _C.MODEL.BODY.with_reid = False

    _C.MODEL.POSE = CN()
    _C.MODEL.POSE.cfg = ''
    _C.MODEL.POSE.weight = ''
    _C.MODEL.POSE.trt_engine = ''
    _C.MODEL.POSE.deploy_cfg = ''

    _C.MODEL.BEHAVIOR = CN()
    _C.MODEL.BEHAVIOR.cfg = ''
    _C.MODEL.BEHAVIOR.weight = ''
    _C.MODEL.BEHAVIOR.trt_engine = ''

    _C.MODEL.FACEID = CN()
    _C.MODEL.FACEID.cfg = ''
    _C.MODEL.FACEID.weight = ''
    _C.MODEL.FACEID.prototype = ''
    _C.MODEL.FACEID.prototype_cache = ''
    _C.MODEL.FACEID.trt_engine = ''

    _C.OUTPUT = CN()
    _C.OUTPUT.path = 'default'

    _C.DATA = CN()
    _C.DATA.label = ''

    _C.freeze()
    config = _C.clone()

    return config


def parser_args():

    parser = ArgumentParser(description='Process video.')
    parser.add_argument('cfg', help='Path to config file'),
    parser.add_argument('video', help='Path to the video file')
    parser.add_argument('target_type', help='Type of target')
    parser.add_argument('--interval', default=3, type=int,
                        help='Interval of recognised behaviour, in seconds')
    parser.add_argument('--trt', default=False,
                        help='Whether to use TRT engine')
    parser.add_argument('--show', default=False,
                        help='Whether to show inferenced frame')
    parser.add_argument('--save_vid', default=False,
                        help='Whether to save inferenced video')
    parser.add_argument('--show_fps', default=False,
                        help='Whether to show inference fps')
    parser.add_argument('--behavior_label',
                        default='./tools/behavior_label.json',
                        type=str,
                        help='Path to behavior label file')
    args = parser.parse_args()
    cfgs = default()
    cfgs.merge_from_file(args.cfg)
    cfgs.freeze()

    return cfgs, args


def main():
    cfgs, args = parser_args()
    # ViP = VideoProcessor(cfgs, args, trt=args.trt)
    # ViP.process_video(show=args.show, save_vid=args.save_vid, show_fps=args.show_fps)
    ViP = VideoProcessor(cfgs, args, trt=False)
    ViP.process_video(show=False, save_vid=False, show_fps=True)


if __name__ == '__main__':
    main()

