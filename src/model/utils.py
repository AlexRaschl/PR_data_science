import os

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import STORED_MODEL_PATH, VIS_PATH, PP_DICT
from src.preprocessing.datamanager import DataManager


def train_test_model(model: BaseEstimator, param_dict, data_dict, pp_dict=PP_DICT, savefig=None,
                     save_model=True):
    X_train, X_test, y_train, y_test = DataManager.load_tts_data(**data_dict)
    log_tf = pp_dict.get('log_tf', True)

    models = []
    if pp_dict:
        if pp_dict.get('std_scale', False):
            with_mean = pp_dict.get('with_mean', True)
            models.append(StandardScaler(with_mean=with_mean))
        n_comp = pp_dict.get('n_components', -1)
        if n_comp > 0:
            models.append(PCA(n_components=n_comp))

    model.set_params(**param_dict)
    models.append(model)

    pipeline = make_pipeline(*models)

    pipeline.fit(X_train, (y_train if not log_tf else DataManager.log_tf(y_train)).iloc[:, 0])

    m_err, r2 = compute_regression_result(pipeline, X_train, X_test, y_train, y_test, log_tf=log_tf)

    if save_model:
        store_pipeline(pipeline, param_dict, data_dict, pp_dict, m_err, r2)

    return pipeline, m_err, r2


def compute_regression_result(fitted_model, X_train, X_test, y_train, y_test, param_dict=None, log_tf=True):
    y_pred = fitted_model.predict(X_test)
    if log_tf:
        y_pred = DataManager.inv_tf(y_pred)
    m_err = mean_absolute_error(y_test, y_pred)
    print(f'Median of predictions: {np.median(y_pred)}')
    print(f'Mean absolute error: {m_err}')

    r2 = None
    score_op = getattr(fitted_model, 'score', None)
    if score_op and callable(score_op):
        r2 = fitted_model.score(X_test, y_test if not log_tf else DataManager.log_tf(y_test).iloc[:, 0])
        print(f'R^2={r2}')

    return m_err, r2


def plot_regression_analysis(y_test, y_pred, savefig=None, title='model'):
    # TODO
    pass


def store_pipeline(pipeline, param_dict, data_dict, pp_dict, m_err=None, r2=None, path=STORED_MODEL_PATH):
    model_name = type(pipeline[-1]).__name__
    folder = os.path.join(path, model_name)
    os.makedirs(folder, exist_ok=True)

    idx = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]) // 2
    fname = f'{model_name}_{idx}'
    dump(pipeline, filename=os.path.join(folder, fname))
    with open(os.path.join(folder, fname + '_meta_inf.txt'), 'w') as meta_inf:
        meta_inf.write('Params:' + param_dict.__repr__() + '\n')
        meta_inf.write('Data:' + data_dict.__repr__() + '\n')
        meta_inf.write('Preprocess:' + pp_dict.__repr__() + '\n')
        meta_inf.write('M_Abs_err: ' + str(m_err))
        if r2:
            meta_inf.write('\nR2: ' + str(r2))


def load_model(path):
    return load(path)


def handle_savefig(savefig=None):
    if not savefig:
        plt.show()
    elif isinstance(savefig, list):
        dir = os.path.dirname(os.path.join(VIS_PATH, *savefig))
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(VIS_PATH, *savefig))
    else:
        plt.savefig(os.path.join(VIS_PATH, savefig))
