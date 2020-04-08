from sklearn.neighbors import KNeighborsRegressor

from src.config import PP_DICT, FULL_DATA_DICT
from src.model.utils import train_test_model

params = {'n_neighbors': 21, 'p': 1, 'weights': 'distance'}
pp_dict = PP_DICT
data_dict = FULL_DATA_DICT

pipeline, m_err, r2 = train_test_model(KNeighborsRegressor(), params, data_dict, pp_dict, save_model=True)
