from sklearn.neural_network import MLPRegressor

from src.config import FULL_DATA_DICT, PP_DICT
from src.model.utils import train_test_model

# Best found params
params = {'hidden_layer_sizes': (500, 200, 100, 50, 10),
          'activation': 'tanh',
          'solver': 'lbfgs',
          'verbose': True,
          'max_iter': 10_000,
          'learning_rate': 'adaptive',
          'tol': 1e-5}

pp_dict = PP_DICT
data_dict = FULL_DATA_DICT

pipeline, m_err, r2 = train_test_model(MLPRegressor(), params, data_dict, pp_dict, save_model=True)
