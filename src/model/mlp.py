from sklearn.neural_network import MLPRegressor

from src.model.utils import train_model
from src.preprocessing.datamanager import DataManager

X_train, X_test, y_train, y_test = DataManager.load_tts_data(duration_ds=True,
                                                             cnn_ds=True,
                                                             color_ds=True,
                                                             face_ds=True,
                                                             cnn_agg=True,
                                                             ohe_cnn=False,
                                                             ohe_color=True,
                                                             n_labels=None)

# TODO GS PARAMS
params = {'hidden_layer_sizes': (100, 100, 50, 50, 10, 2), 'activation': 'tanh', 'solver': 'lbfgs',
          'verbose': True,
          'max_iter': 1_00,
          'warm_start': True,
          'learning_rate': 'adaptive',
          'tol': 1e-5}
pp_dict = {'n_components': 0.95, 'std_scale': True, 'with_mean': True}
data_dict = {'duration_ds': True,
             'cnn_ds': True,
             'color_ds': True,
             'face_ds': True,
             'cnn_agg': True,
             'ohe_cnn': False,
             'ohe_color': True,
             'n_labels': None
             }

pipeline, m_err = train_model(MLPRegressor(), params, data_dict, pp_dict, save_model=True)
