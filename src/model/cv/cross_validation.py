import logging
import os
import time
from pprint import PrettyPrinter

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.config import GRID_SEARCH_LOG_FOLDER, N_JOBS, SEARCH_METRICS, FILE_CREATION_MODE, SPLIT_SEED
from src.model.cfw import load_train_test_split, preprocess_data


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


def perform_grid_search(model, param_grid: dict, log_name: str,
                        cv: int = 5,
                        seed: int = SPLIT_SEED,
                        dataset: str = 'CNN',
                        n_labels: int = 5, **kwargs):
    """
    Performs a Grid Search to find the best parameters based on n fold Cross Validation.
    It will perform an n-fold cross validation on all distinct parameterisations of the the given model passed via the
    parameter grid. Furthermore, the grid search process is logged via the logger module.

    @param model: Model for which the optimal hyperparameters need to be found.
    @param param_grid: Grid specifying the hyperparameter search space
    @param log_name: Name of the logfile to write the outputs to
    @param cv: Number of folds
    @param seed: Train Test split seed to use
    @param dataset: Features to be used, default uses CNN predictions
    @param n_labels: Number of frame classifications to load if feature includes CNN predictions
    @param kwargs: Parameters getting passed to the preprocess function
    """

    X_train, X_test, y_train, y_test = preprocess_data(
        *load_train_test_split(dataset, seed, feature_frame='CNN' in dataset, n_labels=n_labels, **kwargs),
        n_components=kwargs.get('n_components', -1), std_scale=kwargs.get('std_scale', False))
    general_args = {**kwargs, 'n_labels': n_labels, 'dataset': dataset, 'cv': cv, 'seed': seed}
    logger = init_logger(log_name)
    logger.info('STARTING GRID SEARCH')

    for metric in SEARCH_METRICS:
        logger.info(f'OPTIMIZING FOR {metric}')

        gscv = GridSearchCV(model, param_grid, cv=5, verbose=5,
                            scoring=metric, iid=True, n_jobs=N_JOBS)

        gscv.fit(X_train, y_train)

        write_report(gscv, model, logger, general_args)

    logger.info('GRID SEARCH FINISHED!')
    logging.shutdown()


def write_report(gscv, model, logger, general_args: dict):
    logger.info(f'CV_Results:\n'
                f'{PrettyPrinter().pformat(object=gscv.cv_results_)}')
    logger.info('Best parameters set found on training set:\n'
                f'\t{gscv.best_params_}\n\n'
                f'Additional parameters:\n'
                f' \t{PrettyPrinter().pformat(general_args)}')

    means = gscv.cv_results_['mean_test_score']
    stds = gscv.cv_results_['std_test_score']
    res = 'Grid scores on training set:'
    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
        res = f'{res}\n\t{("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))}'
    logger.info(res)

    __test_best_param_setting(gscv, model, logger, general_args)


def __test_best_param_setting(gscv: GridSearchCV, model, logger, general_args):
    logger.info('Detailed classification report:\n'
                'Trained on full train set. Evaluated at full test set!')

    ds = general_args.get('dataset')
    seed = general_args.get('seed')
    n_components = general_args.get('n_components', -1.0)
    std_scale = general_args.get('std_scale', False)

    X_train, X_test, y_train, y_test = preprocess_data(
        *load_train_test_split(ds, seed, feature_frame='CNN' in ds), n_components, std_scale)

    # restore the original parameter dynamically
    # fit on that one to not use over-fitted grid-search
    model.set_params(**gscv.best_params_)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    logger.info(
        'Regression Report:\n'
        f'Mean absolute error: {mean_absolute_error(y_test, predictions)}\n'
        f'Mean Squared error: {mean_squared_error(y_test, predictions)}')
