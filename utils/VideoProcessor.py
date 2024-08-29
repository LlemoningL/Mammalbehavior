import cv2
import time
import collections
import math
from pathlib import Path
from .Visualizer import Visualizer
from .ModelHandler import ModelHandler, ModelTRTHandler
from .DataManager import DataManager
from concurrent.futures import ThreadPoolExecutor
from utils.util import get_color


class VideoProcessor:
    def __init__(self,
                 cfg,
                 arg,
                 trt=False,
                 line_thickness=3,
                 padding=8):

        video_path = Path(arg.video)
        output_root = Path(cfg.OUTPUT.path)
        video_save_path = output_root / video_path.stem
        video_save_path.mkdir(parents=True, exist_ok=True)
        self.save_name = f'{video_save_path}/vis_{arg.interval}s_{video_path.name}'
        self.interval = arg.interval
        self.cap = cv2.VideoCapture(str(video_path))
        self.video_frame_cnt = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_padded_num = int(math.log10(self.video_frame_cnt)) + 1

        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.time_padded_num = int(math.log10(int(self.video_frame_cnt/self.fps))) + 1
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # noqa

        self.Visualizer = Visualizer(cfg, line_thickness=line_thickness, padding=padding)
        self.DataManager = DataManager(
            video_save_path,
            arg.target_type,
            arg.behavior_label,
            arg.interval)
        behavior_label = self.DataManager.label()
        if trt:
            self.ModelHandler = ModelTRTHandler(
                cfg,
                self.size,
                behavior_label,
                self.DataManager,
                self.fps)
        else:
            self.ModelHandler = ModelHandler(
                cfg,
                self.size,
                behavior_label,
                self.DataManager,
                self.fps)
        self.timestamps = collections.deque(maxlen=self.fps)
        # self.frame_queue = queue.Queue(maxsize=300)
        self.write_queue = collections.deque(maxlen=3000)
        self.is_processing = True
        self.executor = ThreadPoolExecutor(max_workers=3)  # 创建线程池
        self.show_fps_ = False
        self.show_frame = False
        self.id_bbox_colors = dict()
        self.id_bbox_colors = self.color([i for i in range(0, 200)])

    def process_video(self, show=False, save_vid=False, show_fps=False):
        print('\n' + '-' * 20)
        self.show_fps_ = show_fps
        self.save_vid = save_vid
        self.show_frame = show
        self.executor.submit(self.process_frames)
        if self.save_vid:
            self.videoWriter = cv2.VideoWriter(
                f'{str(self.save_name)}', self.fourcc, self.fps, self.size)
            self.executor.submit(self.write_frames)
        if self.show_frame or self.save_vid:
            self.executor.submit(self.display_frames)


        # 等待所有任务完成
        self.executor.shutdown(wait=True)
        while self.is_processing and self.write_queue:
            time.sleep(0.01)
        self.cleanup()

    def process_frames(self):
        try:
            time_tag = False
            frame_tag = False
            interval = self.interval
            time_counter = 0
            frame_counter = 0
            recognition_counter = 0
            start_time = time.time()
            while self.cap.isOpened():
                flag, frame = self.cap.read()
                if not flag:
                    break
                current_frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame_id == 1:
                    frame_tag = True
                self.current_frame_time_stamp = round(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                if self.current_frame_time_stamp - time_counter >= interval:
                    recognition_counter += 1
                    if recognition_counter == 1:  # 只在每个间隔的第一次触发
                        time_tag = True
                    time_counter = self.current_frame_time_stamp
                else:
                    recognition_counter = 0

                frame_counter += 1
                if frame_counter >= 5:
                    frame_tag = True
                    frame_counter = 0
                self.ModelHandler.process_frame(
                    frame,
                    frame_tag,
                    time_tag,
                    current_frame_id,
                    self.current_frame_time_stamp)

                if self.show_fps_:
                    vid_fps = self.show_fps(current_frame_id)
                else:
                    vid_fps = 0

                print(f"\rFrame id: {current_frame_id:0{self.frame_padded_num}d}/{self.video_frame_cnt}\t"
                      f"Time stamp: {self.current_frame_time_stamp:0{self.time_padded_num}d}s\t"
                      f"FPS: {vid_fps:.0f} ", end='')
                time_tag = False
                frame_tag = False

            end_time = time.time()
            time_elapsed = end_time - start_time
            avg_fps = self.video_frame_cnt / time_elapsed
            print(f'\nelapsed {time_elapsed:.1f} s, avg {avg_fps:.1f} FPS')
            self.DataManager.save_generated_data()
            if self.show_frame or self.save_vid:
                while self.ModelHandler.processed_frame_qeue:
                    time.sleep(0.01)
            else:
                while self.ModelHandler.processed_frame_qeue:
                    self.ModelHandler.processed_frame_qeue.popleft()
        finally:
            self.is_processing = False
            self.cap.release()

    def show_fps(self, current_frame_id):
        self.timestamps.append(time.time())
        if len(self.timestamps) == self.timestamps.maxlen:
            # 计算最近帧的总时间
            total_time = self.timestamps[-1] - self.timestamps[0]
            # 帧率等于窗口大小除以总时间
            fps = len(self.timestamps) / total_time
        else:
            fps = 0

        return fps

    def cleanup(self):
        # self.cap.release()
        # if hasattr(self, 'VideoWriter'):
        #     self.videoWriter.release()
        cv2.destroyAllWindows()
        print('all done')

    def write_frames(self):
        while self.is_processing or self.write_queue:
            try:
                frame = self.write_queue.popleft()
                self.videoWriter.write(frame)
            except:
                time.sleep(0.01)
                continue

    def display_frames(self):
        while self.is_processing or self.ModelHandler.processed_frame_qeue:
            # try:
                # frame, frame_coordinates, data_sample = self.ModelHandler.get_frame()
                # if frame is None:
                #     time.sleep(0.01)  # 短暂睡眠，避免CPU过度使用
                #     continue
            try:
                frame, frame_coordinates, data_sample = self.ModelHandler.processed_frame_qeue.popleft()
            except:
                time.sleep(0.01)
                continue

            frame = self.Visualizer.visualize(
                frame,
                frame_coordinates,
                self.id_bbox_colors,
                data_sample)
            if self.show_frame:
                self.Visualizer.show(frame)
            if self.save_vid:
                self.write_queue.append(frame)
            # except:
            #    pass

    def color(self, track_ids):
        bbox_colors = get_color(track_ids)
        for t_id, b_colors in zip(track_ids, bbox_colors):
            if t_id not in self.id_bbox_colors:
                self.id_bbox_colors[int(t_id)] = tuple(b_colors)
        return self.id_bbox_colors

