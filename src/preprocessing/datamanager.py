import os
from typing import Dict, List

import pandas as pd

from src.config import *


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def sample_path_from_id(v_id: str):
        return os.path.join(DL_PATH, v_id)

    @staticmethod
    def sample_path_from_dict(vid_dict: Dict):
        return os.path.join(DL_PATH, vid_dict['v_id'])

    @staticmethod
    def extract_v_ids(X: pd.DataFrame) -> List[str]:
        return X.v_id.tolist()
