import cv2
import time
import collections
import queue
from pathlib import Path
from .Visualizer import Visualizer
from .ModelHandler import ModelHandler, ModelTRTHandler
from .DataManager import DataManager
from concurrent.futures import ThreadPoolExecutor


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
            self.ModelHandler = ModelTRTHandler(
                cfg,
                size,
                behavior_label,
                self.DataManager,
                self.Visualizer,
                fps)
        else:
            self.ModelHandler = ModelHandler(
                cfg,
                size,
                behavior_label,
                self.DataManager,
                self.Visualizer,
                fps)
        self.timestamps = collections.deque(maxlen=fps)
        self.frame_queue = queue.Queue(maxsize=300)
        self.write_queue = queue.Queue(maxsize=300)
        self.is_processing = True
        self.executor = ThreadPoolExecutor(max_workers=3)  # 创建线程池
        self.show_fps_ = False
        self.save_vid = False
        self.show_frame = False

    def process_video(self, show=False, save_vid=False, show_fps=False):
        self.show_fps_ = show_fps
        self.save_vid = save_vid
        self.show_frame = show
        self.executor.submit(self.process_frames)
        if self.save_vid:
            self.executor.submit(self.write_frames)
        if self.show_frame:
            self.executor.submit(self.display_frames)

        # 等待所有任务完成
        self.executor.shutdown(wait=True)
        self.cleanup()

    def process_frames(self):
        try:
            print("Processing frames...")
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

                frame = self.ModelHandler.process_frame(frame,
                                                        frame_tag,
                                                        time_tag,
                                                        current_frame_id,
                                                        self.current_frame_time_stamp,)

                if self.show_fps_:
                    self.show_fps(current_frame_id)
                self.frame_queue.put((frame, current_frame_id, self.current_frame_time_stamp))
                time_tag = False
                frame_tag = False

            end_time = time.time()
            time_elapsed = end_time - start_time
            avg_fps = self.video_frame_cnt / time_elapsed
            print(f'elapsed {time_elapsed:.2f} s, avg {avg_fps:.1f} FPS')
            self.DataManager.save_generated_data()
            self.is_processing = False
        except Exception as e:
            print(f"Error in process_frames: {e}")
            self.is_processing = False

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

    def cleanup(self):
        self.cap.release()
        self.videoWriter.release()
        cv2.destroyAllWindows()
        print('all done')

    def write_frames(self):
        while self.is_processing or not self.write_queue.empty():
            try:
                frame = self.write_queue.get(timeout=1)
                self.videoWriter.write(frame)
            except queue.Empty:
                continue

    def display_frames(self):
        while self.is_processing or not self.frame_queue.empty():
            try:
                frame, current_frame_id, timestamp = self.frame_queue.get(timeout=1)
                self.Visualizer.show(frame)
                if self.save_vid:
                    self.write_queue.put(frame)
            except queue.Empty:
                continue


