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
    parser.add_argument('--interval', default=3, type=int, help='Interval to recognize, in seconds')
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
    VP = VideoProcessor(cfgs, args, trt=True)
    # VP = VideoProcessor(cfgs, args, trt=False)
    VP.process_video(show=False, save_vid=False, show_fps=True)
    # VP.process_video(show=True, save_vid=True, show_fps=True)
    VP.cleanup()


if __name__ == '__main__':
    main()

