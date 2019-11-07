from os import unlink

import cv2
from ffmpy import FFmpeg

from src.config import DL_PATH, RES_RSCLD, EMPTY_PATH
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


if __name__ == '__main__':
    vm = VideoFormatter()
    vm.prepare_videos(3)
