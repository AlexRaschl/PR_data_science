import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

from src.config import VIS_PATH
from src.model import cfw


def compute_baseline(model, kw_load: dict, loss_function=mean_absolute_error, savefig=None):
    X_train, X_test, y_train, y_test = cfw.load_train_test_split(**kw_load)

    model.fit(X_train, y_train.iloc[:, 0].ravel())

    y_pred = model.predict(X_test)

    print(f'Median of predictions: {np.median(y_pred)}')
    err = loss_function(y_test.iloc[:, 0].ravel(), y_pred.ravel())
    print(f'{loss_function.__name__}: {err}')

    plot_prediction_results(y_test, y_pred, loss_function, savefig, title=str(model))


def plot_prediction_results(y_test, y_pred, loss_function=mean_absolute_error, savefig=None, title='model'):
    err = loss_function(y_test, y_pred)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5),
                           gridspec_kw={'top': 0.9, 'left': 0.1, 'right': 0.92, 'height_ratios': [1]})
    plt.suptitle(f'Test/Pred of {title}: {loss_function.__name__}={err}')
    ax[0].hist(y_test.iloc[:, 0].ravel(), bins=25, color='red')
    ax[0].set(yscale='log')
    ax[0].set_title('Test set distribution')
    ax[1].hist(y_pred.ravel(), bins=25)
    ax[1].set_title('Prediction distribution')
    ax[1].set(yscale='log')
    ax[2].boxplot(y_test.iloc[:, 0].ravel(), boxprops=dict(color='red'))
    ax[2].set_title('Test set summary statistics')
    ax[2].set_xticks([])
    ax[3].boxplot(y_pred.ravel())
    ax[3].set_title('Predictions summary statistics')
    ax[3].set_xticks([])
    handle_save(savefig)


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, X_test, n=10, savefig=None):
    idx = y_true.argsort(order='asc')
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    X_test = X_test[idx]
    fig = plt.figure(figsize=(10, 10))
    plt.plot(y_true[:n], y_pred[:n])

    handle_save(savefig)


class SimplePredictor:
    def __init__(self, function):
        self.constant = None
        self.function = function

    def fit(self, X_train, y_train):
        # print(self.function(y_train, axis=0))
        self.constant = self.function(y_train, axis=0)

    def predict(self, X_test):
        # print(X_test.shape[0])
        return np.ones(X_test.shape[0]) * self.constant

    def __str__(self):
        return self.function.__name__


def handle_save(savefig: None):
    if not savefig:
        plt.show()
    elif isinstance(savefig, list):
        plt.savefig(os.path.join(VIS_PATH, *savefig))
    else:
        plt.savefig(os.path.join(VIS_PATH, savefig))


def compute_KNN1_baseline(savefig=['baseline', 'baseline_test_pred_comparison.png']):
    X_train, X_test, y_train, y_test = cfw.load_train_test_split(feature_frame=True, one_hot_inputs=True, n_labels=5)

    knn = KNeighborsRegressor(n_neighbors=1)

    knn.fit(X_train, y_train.iloc[:, 0].ravel())

    y_pred = knn.predict(X_test)
    # y_pred = np.array([y_train.median() for i in range(len(y_pred))])

    print(f'Median of predictions: {np.median(y_pred)}')

    m_err = mean_absolute_error(y_test.iloc[:, 0].ravel(), y_pred.ravel())

    print(f'Mean absolute error: {m_err}')
    r2 = knn.score(X_test, y_test.iloc[:, 0].ravel())
    print(f'R^2={r2}')

    fig, ax = plt.subplots(1, 4, figsize=(20, 5),
                           gridspec_kw={'top': 0.9, 'left': 0.1, 'right': 0.92, 'height_ratios': [1]})
    plt.suptitle(f'Test/Pred comparison: R^2={r2}, m_abs={m_err}')
    ax[0].hist(y_test.iloc[:, 0].ravel(), bins=25, color='red')
    ax[0].set(yscale='log')
    ax[0].set_title('Test set distribution')
    ax[1].hist(y_pred.ravel(), bins=25)
    ax[1].set_title('Prediction distribution')
    ax[1].set(yscale='log')
    ax[2].boxplot(y_test.iloc[:, 0].ravel(), boxprops=dict(color='red'))
    ax[2].set_title('Test set summary statistics')
    ax[2].set_xticks([])
    ax[3].boxplot(y_pred.ravel())
    ax[3].set_title('Predictions summary statistics')
    ax[3].set_xticks([])
    handle_save(savefig)


def compute_baseline_for_labels(start=1, end=2, savefig=None):
    knn = KNeighborsRegressor(n_neighbors=1)  # Baseline classifier

    medians = []
    means = []
    errors = []

    for i in range(start, end + 1):
        X_train, X_test, y_train, y_test = cfw.load_train_test_split(feature_frame=True,
                                                                     one_hot_inputs=True,
                                                                     n_labels=i)
        knn.fit(X_train, y_train.iloc[:, 0].ravel())

        y_pred = knn.predict(X_test)
        median = np.median(y_pred, axis=0)
        medians.append(median)
        means.append(np.mean(y_pred, axis=0))
        m_err = mean_absolute_error(y_test.iloc[:, 0].ravel(), y_pred.ravel())
        errors.append(m_err)
        print(f'_____________  N_labels: {i}  ______________')
        print(f'Prediction median: {median}')
        print(f'Mean absolute error: {m_err}\n')

    _, _, _, y_test = cfw.load_train_test_split(feature_frame=True, one_hot_encode=True, n_labels=1)
    fig = plt.figure(figsize=(14, 8))
    plt.title("Baseline predictions for all top-x CNN-labels")
    plt.xlabel("Number of stored top predictions per frame")
    plt.ylabel("")
    plt.plot(np.array(medians), color='yellow', marker='o')
    plt.plot(np.array(means), color='blue', marker='x')
    plt.plot(np.array(errors), color='red')
    plt.plot(np.ones((end - start + 1,), dtype=np.float64) * y_test.mean().values, color='black')
    plt.legend(('Median', 'mean', 'm abs err', 'True mean'))
    plt.xticks(range(start, end + 1))
    handle_save(savefig)


ohe_kw = {'feature_frame': True, 'one_hot_inputs': True, 'n_labels': 5}

compute_baseline(KNeighborsRegressor(n_neighbors=1), kw_load=ohe_kw, loss_function=mean_absolute_error,
                 savefig=['baseline', 'KNN_1.png'])
compute_baseline(SimplePredictor(np.median), kw_load=ohe_kw, loss_function=mean_absolute_error,
                 savefig=['baseline', 'predict_median.png'])
compute_baseline(SimplePredictor(np.mean), kw_load=ohe_kw, loss_function=mean_absolute_error,
                 savefig=['baseline', 'predict_mean.png'])

compute_baseline_for_labels(end=10, savefig=['baseline', 'KNN_1_label_comparison.png'])
