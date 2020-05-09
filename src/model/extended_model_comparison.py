import src.model.utils as ut
from src.preprocessing.datamanager import DataManager
from src.config import FULL_DATA_DICT, N_JOBS
from src.model.cfw import mean_absolute_percentage_error, mean_relative_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

models_to_load = ['.\models\KNeighborsRegressor\KNeighborsRegressor_FINAL',
                  '.\models\MLPRegressor\MLPRegressor_FINAL',
                  '.\models\RandomForestRegressor\RandomForestRegressor_FINAL',
                  '.\models\SVR\SVR_FINAL']
model_specs = [
    {'n_neighbors': 21, 'p': 1, 'weights': 'distance'},
    {'hidden_layer_sizes': (500, 200, 100, 50, 10),
     'activation': 'tanh',
     'solver': 'lbfgs',
     'verbose': True,
     'max_iter': 10_000,
     'learning_rate': 'adaptive',
     'tol': 1e-5},
    {
        'n_jobs': N_JOBS,
        'criterion': 'mse',
        'max_depth': 25,
        'max_features': 'auto',
        'n_estimators': 200,
        'max_samples': 0.75,
        'ccp_alpha': 0.0
    },
    {'C': 1,
     'epsilon': 0.01,
     'kernel': 'rbf',
     'shrinking': True,
     'tol': 0.001}
]
save_paths_resid_comp = [['fitted_models', 'KNN_MAPE_plot'],
                         ['fitted_models', 'MLP_MAPE_plot'],
                         ['fitted_models', 'RF_MAPE_plot'],
                         ['fitted_models', 'SVR_MAPE_plot']]

save_path_overall = ['fitted_models', 'FinalModelComparison']


# TODO: MAPE, MRE, sorted plots
def check_log_tf(path):
    meta_file = path + '_meta_inf.txt'
    with open(meta_file, 'r') as f:
        return '\'log_tf\': True' in f.read()


def generate_model_descr(pipeline, model_info: dict) -> str:
    model_name = pipeline[-1].__class__.__name__
    return f'{model_name + " " + str(model_info) if model_info else model_name}'


def plot_sorted_residuals(path, data_dict=FULL_DATA_DICT, model_info=None, savefig=None, log_tf=True):
    pipeline = ut.load_model(path)
    X_train, X_test, y_train, y_test = DataManager.load_tts_data(**data_dict)
    y_pred = pipeline.predict(X_test)
    if check_log_tf(path):
        y_pred = DataManager.inv_tf(y_pred)

    y_test = np.array(y_test).ravel()

    sorted_idx = y_test.argsort()
    y_test_srt = y_test[sorted_idx]
    y_pred_srt = y_pred[sorted_idx]

    if log_tf:
        y_test_srt, y_pred_srt = DataManager.log_tf(y_test_srt, y_pred_srt)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10),
                                        gridspec_kw={'top': 0.90})

    x = range(len(y_test_srt))
    plt.suptitle('Sorted Viewcount error comparison:\n' +
                 'MAPE: %.3f\n'.format(mean_absolute_percentage_error(y_test, y_pred)) +
                 generate_model_descr(pipeline, model_info))
    plt.tight_layout()
    ax1.scatter(x=x, y=y_test_srt, s=1.5, alpha=0.9, marker='x')
    ax2.scatter(x=x, y=y_pred_srt, s=1.5, alpha=0.9, marker='x')
    ax2.set_ylabel('log(viewcounts)' if log_tf else 'viewcounts')
    ax1.set_ylabel('log(viewcounts)' if log_tf else 'viewcounts')
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xticks([])
    ax1.set_xticks([])

    rel_error = np.abs((y_test_srt - y_pred_srt) / y_test_srt) * 100
    ax3.scatter(x=x, y=rel_error, s=1.5, alpha=0.9, marker='x', c='red')
    ax3.set_ylim((0.0, 100.00))
    ax3.set_ylabel('Absolute Percentage Error')
    ax3.set_xlabel('Sorted Sample Index')
    ut.handle_savefig(savefig)

    return mean_absolute_percentage_error(y_test, y_pred)


def compute_sorted_residual_plots_for_all(models_to_load, model_specs, save_paths):
    for model, model_info, save_path in zip(models_to_load, model_specs, save_paths):
        plot_sorted_residuals(model, data_dict=FULL_DATA_DICT, model_info=model_info, savefig=save_path)


def compute_scores(model_path, X_test, y_test):
    pipeline = ut.load_model(model_path)
    y_pred = pipeline.predict(X_test)

    if check_log_tf(model_path):
        y_pred = DataManager.inv_tf(y_pred)

    y_test_vec = np.array(y_test).ravel()

    mae = mean_absolute_error(y_test_vec, y_pred)
    mre = mean_relative_error(y_test_vec, y_pred)
    mape = mean_absolute_percentage_error(y_test_vec, y_pred)

    return pipeline[-1].__class__.__name__, mae, mre, mape


def error_summary(models_to_load, model_specs, save_path, ylimr, data_dict=FULL_DATA_DICT):
    def populate_axes(ax, df, col: str):
        bars = sns.barplot(x=df.index, y=df[col], ax=ax, alpha=0.8)
        for p in bars.patches:
            bars.annotate(format(int(p.get_height()), ',d'), (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center',
                          va='center', xytext=(0, 10), textcoords='offset points')

    X_train, X_test, y_train, y_test = DataManager.load_tts_data(**data_dict)
    scores = [compute_scores(model_path, X_test, y_test) for model_path in models_to_load]

    df = pd.DataFrame(scores)
    df.columns = ['Model', 'MAE', 'MRE', 'MAPE']
    df.set_index('Model', inplace=True)

    sns.set_style(style='whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), gridspec_kw={'top': 0.92})

    plt.suptitle("Final Model Comparison")

    populate_axes(ax1, df, 'MAE')
    populate_axes(ax2, df, 'MRE')
    populate_axes(ax3, df, 'MAPE')

    ut.handle_savefig(save_path)


error_summary(models_to_load, model_specs, save_path_overall, [0, 1000])
# compute_sorted_residual_plots_for_all(models_to_load, model_specs, save_paths_resid_comp)
