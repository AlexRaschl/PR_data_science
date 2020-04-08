import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

from src.config import BL_DATA_DICT
from src.model.utils import handle_savefig
from src.preprocessing.datamanager import DataManager


def compute_baseline(model, kw_load: dict = BL_DATA_DICT, loss_function=mean_absolute_error, savefig=None, log_tf=True,
                     title='model'):
    X_train, X_test, y_train, y_test = DataManager.load_tts_data(**kw_load)

    if log_tf:
        y_train = DataManager.log_tf(y_train)  # Log transform of response

    model.fit(X_train, y_train.iloc[:, 0].ravel())

    y_pred = model.predict(X_test)

    if log_tf:
        y_pred = DataManager.inv_tf(y_pred)  # Transform prediction results back

    err = loss_function(y_test.iloc[:, 0].ravel(), y_pred.ravel())

    # print(f'Median of predictions: {np.median(y_pred)}')
    # print(f'{loss_function.__name__}: {err}')

    plot_prediction_results(y_test, y_pred, loss_function, savefig, title=title)
    return y_pred


def plot_prediction_results(y_test, y_pred, loss_function=mean_absolute_error, savefig=None, title='model'):
    err = loss_function(y_test, y_pred)

    fig, ax = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'hspace': 0.2})
    plt.suptitle(f'Test/Pred of {title}: {loss_function.__name__}={err}')
    ax[0, 0].hist(y_test.iloc[:, 0].ravel(), bins=25, color='red')
    ax[0, 0].set_title('Test set distribution')
    ax[0, 1].hist(y_pred.ravel(), bins=25)
    ax[0, 0].set_yscale('log')
    ax[0, 1].set_yscale('log')
    ax[0, 0].set_ylabel('count (log)')
    ax[0, 1].set_ylabel('count (log)')
    ax[0, 1].set_yticks(ax[0, 0].get_yticks())
    ax[0, 1].set_ylim(ax[0, 0].get_ylim())

    ax[0, 1].set_title('Prediction distribution')
    ax[0, 2].boxplot(y_test.iloc[:, 0].ravel(), boxprops=dict(color='red'))
    ax[0, 2].set_title('Test set summary statistics')
    ax[0, 2].set_xticks([])
    ax[0, 3].boxplot(y_pred.ravel())
    ax[0, 3].set_title('Predictions summary statistics')
    ax[0, 3].set_xticks([])
    ax[0, 3].set_yticks(ax[0, 2].get_yticks())
    ax[0, 3].set_ylim(ax[0, 2].get_ylim())

    ## Log transforms
    y_test = y_test.apply(np.log10)
    y_pred = np.log10(y_pred)
    ax[1, 0].hist(y_test.iloc[:, 0].ravel(), bins=25, color='red')
    ax[1, 0].set_title('Test set distribution (log10)')
    ax[1, 1].hist(y_pred.ravel(), bins=25)
    ax[1, 1].set_xticks(ax[1, 0].get_xticks())
    ax[1, 1].set_xlim(ax[1, 0].get_xlim())
    ax[1, 0].set_ylabel('count')
    ax[1, 1].set_ylabel('count')

    ax[1, 1].set_title('Prediction distribution (log10)')
    ax[1, 2].boxplot(y_test.iloc[:, 0].ravel(), boxprops=dict(color='red'))
    ax[1, 2].set_title('Test set summary statistics (log10)')
    ax[1, 2].set_xticks([])
    ax[1, 3].boxplot(y_pred.ravel())
    ax[1, 3].set_title('Predictions summary statistics (log10)')
    ax[1, 3].set_xticks([])
    ax[1, 0].hist(y_test.iloc[:, 0].ravel(), bins=25, color='red')
    ax[1, 3].set_yticks(ax[1, 2].get_yticks())
    ax[1, 3].set_ylim(ax[1, 2].get_ylim())

    handle_savefig(savefig)


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


def compute_KNN1_baseline(savefig=['baseline', 'baseline_test_pred_comparison.png'], log_tf=True):
    X_train, X_test, y_train, y_test = DataManager.load_tts_data(**BL_DATA_DICT)

    if log_tf:
        y_train = y_train.apply(np.log10)
        y_test = y_test.apply(np.log10)

    knn = KNeighborsRegressor(n_neighbors=1)

    knn.fit(X_train, y_train.iloc[:, 0].ravel())

    y_pred = knn.predict(X_test)

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
    handle_savefig(savefig)


def compute_baseline_for_labels(start=1, end=2, savefig=None, log_tf=False):
    knn = KNeighborsRegressor(n_neighbors=1)  # Baseline classifier

    medians = []
    means = []
    errors = []

    for i in range(start, end + 1):
        X_train, X_test, y_train, y_test = DataManager.load_tts_data(cnn_ds=True, cnn_agg=False, ohe_cnn=True,
                                                                     n_labels=i)  # TODO possibly faulty

        if log_tf:
            y_train = y_train.apply(np.log10)
            y_test = y_test.apply(np.log10)

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

    _, _, _, y_test = DataManager.load_tts_data(cnn_agg=False, ohe_cnn=True, n_labels=1)
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
    handle_savefig(savefig)


#
# compute_baseline(KNeighborsRegressor(n_neighbors=1), kw_load=BL_KW, loss_function=mean_absolute_error,
#                  savefig=['baseline', 'KNN_1_baseline.png'], log_tf=False, title='KNN-Regressor')
# compute_baseline(SimplePredictor(np.median), kw_load=BL_KW, loss_function=mean_absolute_error,
#                  savefig=['baseline', 'predict_median_baseline.png'], log_tf=True, title='Constant Median')
# compute_baseline(SimplePredictor(np.mean), kw_load=BL_KW, loss_function=mean_absolute_error,
#                  savefig=['baseline', 'predict_mean_baseline.png'], log_tf=True, title='Constant Mean')

# compute_baseline_for_labels(end=10, savefig=['baseline', 'KNN_1_label_comparison.png'])


def plot_predictions(y_true, y_pred, n=10, savefig=None):
    idx = y_true.ravel().argsort(axis=0)
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(x=y_true[:n], y=y_pred[:n], marker='x', color='black')
    plt.xlabel('True viewcounts')
    plt.ylabel('Predicted viewcounts')
    plt.yscale(plt.gca().get_xscale())
    plt.yticks(plt.gca().get_xticks())
    plt.ylim(plt.gca().get_xlim())

    handle_savefig(savefig)

# X_train, X_test, y_train, y_test = DataManager.load_tts_data(color_ds=False, ohe_color=True, face_ds=True, duration_ds=True, cnn_agg=True)
#
# mms = StandardScaler()
# X_train = mms.fit_transform(X_train.values)
# X_test = mms.transform(X_test.values)
#
# y_train = y_train.apply(np.log10)
# y_test = y_test.apply(np.log10)
#
# model = KNeighborsRegressor(3)
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# plot_predictions(y_test.values.ravel(), y_pred.ravel(), n=10000)
# m_err = mean_absolute_error(y_test.values.ravel(), y_pred.ravel())
#
# plt.figure(figsize=(10, 10))
# sns.kdeplot(y_test.values.ravel())
#
# plt.show()
