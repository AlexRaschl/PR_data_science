from os import *

import cv2
from ffmpy import FFmpeg
from skimage.measure._structural_similarity import structural_similarity as ssim

from src.config import DL_PATH, RES_RSCLD, EMPTY_PATH, SAMPLE_OFFSET, N_SAMPLES, SIMILARITY_THRES, N_CRAWLS, FAIL_PATH
from src.config import FFMPEG_PATH
from src.database.db_utils import get_collection_from_db


class VideoFormatter:
    def __init__(self):
        self.collection = get_collection_from_db()
        self.resolution = RES_RSCLD

    def prepare_videos(self, num, delete_old=True):

        for i in range(num):
            print(f"Preprocessing-call Nr: {i + 1}")
            vid = self.collection.find_one({"$and": [{'v_found': True}, {'v_filepath':
                                                                             {"$ne": EMPTY_PATH}},
                                                     {'v_width': {"$ne": RES_RSCLD[0]}},
                                                     {'v_height': {"$ne": RES_RSCLD[1]}}]})
            try:
                self.reformat(vid)
                # self.compress(vid)
            except Exception as e:
                print(e)
                return

    def compress(self, vid, delete_old=True):
        vid_name = vid['v_title']
        old_path = f"{DL_PATH}{vid['v_filepath']}"

        new_path = f"{DL_PATH}{vid_name}.mp4"

        inputs = {old_path: None}
        outputs = {new_path: '-vcodec libx264 -crf 20 -y'}

        try:
            ff = FFmpeg(FFMPEG_PATH, inputs=inputs, outputs=outputs)
            print(ff.cmd)
            ff.run()
        except Exception as e:
            print(e)
        else:
            print(f"Updating DB rescaled: {vid_name}")
            self.collection.update_one({'v_id': vid['v_id']}, {"$set": {'v_filepath': f"{vid_name}.mp4",
                                                                        'v_compressed': True
                                                                        }})
            if delete_old:
                unlink(old_path)

    def reformat(self, vid, delete_old=True):
        old_path = vid['v_filepath']

        if vid['v_res'] == RES_RSCLD:
            print("No rescale needed!")
            return

        cap = cv2.VideoCapture(f"{DL_PATH}{old_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        filepath = f"{DL_PATH}{vid['v_title']}-rescaled.mp4"
        out = cv2.VideoWriter(filepath, fourcc, fps, RES_RSCLD)

        while True:
            ret, frame = cap.read()
            if ret:
                b = cv2.resize(frame, RES_RSCLD, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                out.write(b)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Updating DB reformat: {old_path}")

        self.collection.update_one({'v_id': vid['v_id']},
                                   {"$set":
                                       {
                                           'v_filepath': f"{vid['v_title']}-rescaled.mp4",
                                           'v_width': RES_RSCLD[0],
                                           'v_height': RES_RSCLD[1],
                                           'v_res': RES_RSCLD
                                       }})
        if delete_old:
            unlink(f"{DL_PATH}{old_path}")


class VideoSampler:

    def __init__(self):
        self.collection = get_collection_from_db()
        self.resolution = RES_RSCLD

    def sample(self, n_vids=1000, n_samples=N_SAMPLES, offset=SAMPLE_OFFSET):
        for i in range(n_vids):
            vid = self.collection.find_one({"$and": [
                {'v_found': True},
                {'v_filepath': {"$ne": EMPTY_PATH}},
                {'v_filepath': {"$ne": FAIL_PATH}},
                {'sampled': {"$ne": True}},
            ]})

            if vid is None:
                print("No videos left to sample!")
                return
            print(f'Sampling-step {i}: {vid["v_id"]} - {vid["v_title"]}')
            self.sample_images(vid, n_samples, offset)

    def sample_images(self, vid, n_samples, offset, recheck_mv=True):
        # vid = self.collection.find_one({"$and": [{'v_id': v_id},
        #                                          # {'v_found': True},
        #                                          {'v_filepath':
        #                                               {"$ne": EMPTY_PATH}}]})

        scale_vid = vid['v_res'] != RES_RSCLD
        vid_path = vid['v_filepath']

        cap = cv2.VideoCapture(f"{DL_PATH}{vid_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        filepath = self.sample_path(vid)
        # print(filepath)
        try:
            mkdir(filepath)

            step_size = int(n_frames / n_samples) if offset == 0 \
                else int((n_frames - offset * fps) / n_samples)

            old_frame = None
            equality_count = count = 0
            for i in range(n_frames):  # TODO ? - n_frames % n_samples):
                ret, frame = cap.read()

                if not ret:
                    break

                if i > offset * fps and i % step_size == 0:

                    frame = cv2.resize(frame, RES_RSCLD, fx=0, fy=0,
                                       interpolation=cv2.INTER_CUBIC) if scale_vid else frame
                    # cv2.imshow('image', frame)
                    # cv2.waitKey(1000)
                    cv2.imwrite(path.join(filepath, f'frame{count}.jpg'), frame)
                    count += 1
                    if old_frame is not None:
                        equality_count = equality_count + 1 if self.check_equal(old_frame, frame) else equality_count
                    old_frame = frame

            print(equality_count)
            cap.release()
            cv2.destroyAllWindows()

        except FileExistsError:
            print("Directory already exists!")
            return
        else:
            is_equal = equality_count >= n_samples / 2
            print(is_equal)
            self.collection.update_one({'v_id': vid['v_id']},
                                       {"$set":
                                           {
                                               'sampled': True,
                                               'n_samples': n_samples,
                                               'v_found': not is_equal or not recheck_mv
                                           }})

    @staticmethod
    def sample_path(vid):
        return f"{DL_PATH}{vid['v_id']}"

    def check_equal(self, old_frame, frame):
        diff = ssim(old_frame, frame, data_range=frame.max() - frame.min(), multichannel=True)
        # print(diff)
        return diff > SIMILARITY_THRES


if __name__ == '__main__':
    vm = VideoSampler()
    vm.sample(n_vids=N_CRAWLS)
