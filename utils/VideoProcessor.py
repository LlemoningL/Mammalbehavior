import cv2
import time
import collections
import math
import datetime
import queue
import imageio.v2 as imageio
from pathlib import Path
from .Visualizer import Visualizer
from .ModelHandler import ModelHandler, ModelTRTHandler
from .DataManager import DataManager
from concurrent.futures import ThreadPoolExecutor
from utils.util import get_color
from logger import create_logger


class VideoProcessor:
    def __init__(self,
                 cfg,
                 arg,
                 trt=False,
                 interval=1,
                 line_thickness=3,
                 padding=8,
                 max_queue_len=300):

        video_path = Path(arg.video)
        output_root = Path(cfg.OUTPUT.path)
        video_save_path = output_root / video_path.stem
        video_save_path.mkdir(parents=True, exist_ok=True)
        self.log_output = video_save_path / f'log_{interval}s'
        self.log_output.mkdir(parents=True, exist_ok=True)
        self.save_name = f'{video_save_path}/vis_{interval}s_{video_path.name}'
        self.interval = interval
        self.cap = cv2.VideoCapture(str(video_path))
        self.video_frame_cnt = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_padded_num = int(math.log10(self.video_frame_cnt)) + 1

        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.time_padded_num = int(math.log10(int(self.video_frame_cnt/self.fps))) + 1
        self.total_time = str(datetime.timedelta(seconds=(int(self.video_frame_cnt/self.fps))))
        self.logger_p = create_logger(str(self.log_output), name='P', log_name='process_frames')
        self.logger_p.info('Thread processes frames start...')

        self.Visualizer = Visualizer(cfg, line_thickness=line_thickness, padding=padding)
        self.DataManager = DataManager(
            video_save_path,
            arg.target_type,
            arg.behavior_label,
            interval)
        behavior_label = self.DataManager.label()
        if trt:
            self.ModelHandler = ModelTRTHandler(
                cfg,
                self.size,
                behavior_label,
                self.DataManager,
                self.fps,
                self.logger_p
            )
        else:
            self.ModelHandler = ModelHandler(
                cfg,
                self.size,
                behavior_label,
                self.DataManager,
                self.fps,
                self.logger_p
            )
        self.timestamps = collections.deque(maxlen=self.fps)
        self.processed_frame_qeue = queue.Queue(maxsize=max_queue_len)
        self.write_queue = queue.Queue(maxsize=max_queue_len)
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
        if self.show_frame or self.save_vid:
            self.logger_d = create_logger(str(self.log_output), name='D', log_name='display_frames')
            self.executor.submit(self.display_frames)
            self.logger_d.info('Thread display frames start...')
        if self.save_vid:
            self.videoWriter = imageio.get_writer(
                str(self.save_name),
                format='FFMPEG',
                fps=self.fps,
                macro_block_size=1,
                codec='libx264',
                mode='I',
                pixelformat='yuv420p'
            )
            self.logger_w = create_logger(str(self.log_output), name='W', log_name='write_frames')
            self.executor.submit(self.write_frames)
            self.logger_w.info('Thread write frames start...')

        # 等待所有任务完成
        self.executor.shutdown(wait=True)
        self.logger_p.info('All thread finish')
        while self.is_processing and not self.write_queue.empty():
            time.sleep(0.01)
        self.logger_p.info('Wating for video writing')
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
                _time_in_console = str(datetime.timedelta(seconds=self.current_frame_time_stamp))

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
                frame, frame_coordinates, data_sample = self.ModelHandler.process_frame(
                    frame,
                    frame_tag,
                    time_tag,
                    current_frame_id,
                    self.current_frame_time_stamp)
                self.processed_frame_qeue.put((frame, frame_coordinates, data_sample))

                if self.show_fps_:
                    vid_fps = self.show_fps()
                else:
                    vid_fps = 0

                print(f"\rFrame id: {current_frame_id:0{self.frame_padded_num}d}/{self.video_frame_cnt}\t"
                      f"Time stamp: {_time_in_console}/{self.total_time}\t"
                      f"FPS: {vid_fps:.0f} ", end='')
                self.logger_p.info(f"Frame id: {current_frame_id:0{self.frame_padded_num}d}/{self.video_frame_cnt}\t"
                                   f"Time stamp: {_time_in_console}/{self.total_time}\t"
                                   f"FPS: {vid_fps:.0f} ")
                time_tag = False
                frame_tag = False

            end_time = time.time()
            time_elapsed = round(end_time - start_time)
            _time_elapsed = str(datetime.timedelta(seconds=time_elapsed))

            avg_fps = self.video_frame_cnt / time_elapsed
            print(f'\nelapsed {_time_elapsed}, avg {avg_fps:.1f} FPS')
            self.logger_p.info(f'elapsed {_time_elapsed}, avg {avg_fps:.1f} FPS')
            self.DataManager.save_generated_data()
            self.DataManager.save_data_split()
            if self.show_frame or self.save_vid:
                while not self.processed_frame_qeue.empty():
                    # self.logger_p.info(f'Wating for processed_frame_qeue empty, '
                    #                    f'{self.processed_frame_qeue.qsize()} left')
                    time.sleep(0.01)
            else:
                while not self.processed_frame_qeue.empty():
                    self.processed_frame_qeue.get()
                    # self.logger_p.info(f'Wating for processed_frame_qeue empty, '
                    #                    f'{self.processed_frame_qeue.qsize()} left')
        finally:
            self.logger_p.info(f'All frames processed')
            self.is_processing = False
            self.logger_p(f'self.is_processing [{self.is_processing}]')
            self.cap.release()

    def show_fps(self):
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
        cv2.destroyAllWindows()
        print('all done')
        self.logger_p.info('all done')
        print('-' * 20 + '\n')

    def write_frames(self):
        while self.is_processing or not self.write_queue.empty():
            try:
                frame = self.write_queue.get()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.videoWriter.append_data(frame)
                self.logger_w.info(f'Writing frame, {self.write_queue.qsize()} left')

            except queue.Empty:
                time.sleep(0.01)
                self.logger_w.info(f'Writing for frame')
                continue

    def display_frames(self):
        while self.is_processing or not self.processed_frame_qeue.empty():
            try:
                frame, frame_coordinates, data_sample = self.processed_frame_qeue.get()
            except queue.Empty:
                time.sleep(0.01)
                continue

            frame = self.Visualizer.visualize(
                frame,
                frame_coordinates,
                self.id_bbox_colors,
                data_sample)
            if self.show_frame:
                self.Visualizer.show(frame)
                self.logger_d.info(f'Displayed frame, {self.processed_frame_qeue.qsize()} left')
            if self.save_vid:
                self.write_queue.put(frame)
                self.logger_w.info(f'Add frame')

    def color(self, track_ids):
        bbox_colors = get_color(track_ids)
        for t_id, b_colors in zip(track_ids, bbox_colors):
            if t_id not in self.id_bbox_colors:
                self.id_bbox_colors[int(t_id)] = tuple(b_colors)
        return self.id_bbox_colors

