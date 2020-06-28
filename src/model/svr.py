from sklearn.svm import SVR

from src.config import PP_DICT, FULL_DATA_DICT
from src.model.utils import train_test_model

# Optimal GS Params
params = {'C': 1,
          'epsilon': 0.2,
          'kernel': 'rbf',
          'shrinking': True,
          'tol': 0.001}

pp_dict = PP_DICT
data_dict = FULL_DATA_DICT

pipeline, m_err, r2 = train_test_model(SVR(), params, data_dict, pp_dict, save_model=True)
