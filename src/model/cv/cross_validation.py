import logging
import os
import time
from pprint import PrettyPrinter

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from src.config import GRID_SEARCH_LOG_FOLDER, N_JOBS, SEARCH_METRICS, FILE_CREATION_MODE
from src.model.cfw import load_train_test_split


def init_logger(log_name: str, level: str = 'INFO') -> logging.Logger:
    def suffix_date(file_name: str):
        i_ext = file_name.rfind('.')
        timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')

        # if no file extension was found
        if i_ext < 0:
            return file_name + timestamp + '.log'
        # valid file extension found
        return file_name[:i_ext] + timestamp + file_name[i_ext:]

    os.makedirs(GRID_SEARCH_LOG_FOLDER, exist_ok=True)

    logger = logging.getLogger('gscv.main')

    # remove all previous loggers for this scope
    logger.handlers = []
    logger.setLevel(level)

    # create handle for logfile
    log_file_handler = logging.FileHandler(filename=os.path.join(GRID_SEARCH_LOG_FOLDER
                                                                 , suffix_date(log_name)),
                                           mode=FILE_CREATION_MODE)
    logger.addHandler(log_file_handler)

    # create handle for stdout
    stdout_log_handler = logging.StreamHandler()
    logger.addHandler(stdout_log_handler)

    # set formatter for stdout and logfile
    formatter = logging.Formatter('________________[%(asctime)s]________________\n%(message)s\n')
    log_file_handler.setFormatter(formatter)
    stdout_log_handler.setFormatter(formatter)

    return logger


def preprocess_data(n_components, std_scale: bool = True):
    # TODO
    pass


def perform_grid_search(model, param_grid: dict, log_name: str,
                        cv: int = 5, **kwargs):
    X_train, X_test, y_train, y_test = load_train_test_split()  # TODO PREPROCESS

    logger = init_logger(log_name)
    logger.info('STARTING GRID SEARCH')

    for metric in SEARCH_METRICS:
        logger.info(f'OPTIMIZING FOR {metric}')

        gscv = GridSearchCV(model, param_grid, cv=5, verbose=100,
                            scoring=metric, iid=True, n_jobs=N_JOBS)

        gscv.fit(X_train, y_train)

        # write_report(gscv, classifier, logger)

    logger.info('GRID SEARCH FINISHED!')
    logging.shutdown()


def write_report(gscv, model, logger):
    logger.info(f'CV_Results:\n'
                f'{PrettyPrinter().pformat(object=gscv.cv_results_)}')
    logger.info('Best parameters set found on training set:\n'
                f'\t{gscv.best_params_}')

    means = gscv.cv_results_['mean_test_score']
    stds = gscv.cv_results_['std_test_score']
    res = 'Grid scores on training set:'
    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
        res = f'{res}\n\t{("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))}'
    logger.info(res)

    __test_best_param_setting()


def __test_best_param_setting(gscv: GridSearchCV, model, logger, ):
    logger.info('Detailed classification report:\n'
                'Trained on full train set. Evaluated at full test set!')
    X_train, X_test, y_train, y_test = load_train_test_split()  # TODO PREPROCESS

    # restore the original parameter dynamically
    # fit on that one to not use over-fitted grid-search
    model.set_params(**gscv.best_params_)
    model.fit(X_train, y_train)
    logger.info(
        'Classification Report:\n'
        f'{classification_report(y_test, model.predict(X_test))}')
