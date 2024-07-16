import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.xcb.warning=false"
import cv2
import time
import collections
import threading
import queue
from pathlib import Path
from .Visualizer import Visualizer
from .ModelHandler import ModelHandler, ModelTRTHandler
from .DataManager import DataManager


class VideoProcessor:
    def __init__(self,
                 cfg,
                 arg,
                 trt=False):

        video_path = Path(arg.video)
        output_root = Path(cfg.OUTPUT.path)
        video_save_path = output_root / video_path.stem
        video_save_path.mkdir(parents=True, exist_ok=True)
        save_name = f'{video_save_path}/vis_{video_path.name}'
        self.interval = arg.interval
        self.cap = cv2.VideoCapture(str(video_path))
        self.video_frame_cnt = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # noqa
        self.videoWriter = cv2.VideoWriter(f'{str(save_name)}', fourcc, fps, size)
        self.Visualizer = Visualizer(cfg)
        self.DataManager = DataManager(video_save_path,
                                       arg.target_type,
                                       arg.behavior_label)
        behavior_label = self.DataManager.label()
        if trt:
            self.ModelHandler = ModelTRTHandler(cfg,
                                                size,
                                                behavior_label,
                                                self.DataManager,
                                                self.Visualizer)
        else:
            self.ModelHandler = ModelHandler(cfg,
                                             size,
                                             behavior_label,
                                             self.DataManager,
                                             self.Visualizer)
        self.timestamps = collections.deque(maxlen=fps)
        self.frame_queue = queue.Queue(maxsize=100)
        self.write_queue = queue.Queue(maxsize=100)
        self.show_queue = queue.Queue(maxsize=30)
        self.is_processing = True

    def process_video(self, show=False, save_vid=False, show_fps=False):

        write_thread = threading.Thread(target=self.write_frame)
        show_thread = threading.Thread(target=self.show_frame)
        time_tag = False
        frame_tag = False
        interval = self.interval
        time_counter = 0
        frame_counter = 0
        recognition_counter = 0
        start_time = time.time()
        if save_vid:
            write_thread.start()
        if show:
            show_thread.start()
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
            if frame_counter >= 7:
                frame_tag = True
                frame_counter = 0

            frame = self.ModelHandler.process_frame(frame,
                                                    frame_tag,
                                                    time_tag,
                                                    current_frame_id,
                                                    self.current_frame_time_stamp,
                                                    show_frame=show)

            if show_fps:
                self.show_fps(current_frame_id)
            if save_vid or show:
                self.frame_queue.put(frame)
            time_tag = False
            frame_tag = False

        end_time = time.time()
        time_elapsed = end_time - start_time
        avg_fps = self.video_frame_cnt / time_elapsed
        print(f'elapsed {time_elapsed:.2f}, avg {avg_fps:.2f}')
        self.DataManager.save_generated_data()
        self.cleanup()

    def save_video(self, frame):
        self.videoWriter.write(frame)

    def show_fps(self, current_frame_id):
        self.timestamps.append(time.time())
        if len(self.timestamps) == self.timestamps.maxlen:
            # 计算最近帧的总时间
            total_time = self.timestamps[-1] - self.timestamps[0]
            # 帧率等于窗口大小除以总时间
            fps = len(self.timestamps) / total_time
            print(f"当前帧率: {fps:.1f} FPS, "
                  f"当前帧{current_frame_id}/{self.video_frame_cnt} f, ",
                  f"当前时间{self.current_frame_time_stamp} s")

    def write_frame(self):
        while self.is_processing:
            if not self.write_queue.empty():
                frame = self.write_queue.get()
                self.save_video(frame)
            else:
                time.sleep(0.001)

    def show_frame(self):
        while self.is_processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.Visualizer.show(frame)
            else:
                time.sleep(0.001)

    def cleanup(self):
        self.is_processing = False
        if hasattr(self, 'write_thread'):
            print(f'waiting for write video')
            self.write_thread.join()
            print(f'video saved')
        if hasattr(self, 'show_thread'):
            self.show_thread.join()
        self.cap.release()
        self.videoWriter.release()
        cv2.destroyAllWindows()


