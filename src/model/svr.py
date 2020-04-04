from sklearn.svm import SVR

from src.model.utils import train_model

# TODO GS PARAMS
params = {'degree': 10, 'kernel': 'linear', 'C': 2, 'epsilon': 0.05}  # Optimal GS params
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

pipeline, m_err = train_model(SVR(), params, data_dict, pp_dict, save_model=True)
