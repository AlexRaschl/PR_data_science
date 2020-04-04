from sklearn import ensemble

from src.model.utils import train_model

# TODO GS PARAMS
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

pp_dict = {'n_components': -1}
data_dict = {'duration_ds': True,
             'cnn_ds': False,
             'color_ds': True,
             'face_ds': True,
             'cnn_agg': False,
             'ohe_cnn': False,
             'ohe_color': False,
             'n_labels': None
             }

pipeline, m_err = train_model(ensemble.GradientBoostingRegressor(), params, data_dict, pp_dict, save_model=True)
