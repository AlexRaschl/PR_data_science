from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.config import PP_DICT, FULL_DATA_DICT
from src.model.cv import cross_validation as gscv


def grid_search_knn(grid=None, data_dict: dict = FULL_DATA_DICT, pp_dict: dict = PP_DICT):
    param_grid = grid or [
        {
            'n_neighbors': range(1, 50, 5),
            'weights': ('uniform', 'distance'),
            'p': (2, 1, 1.5)
        }
    ]
    gscv.perform_grid_search(KNeighborsRegressor(), param_grid, data_dict, pp_dict, f'gs_knn.log')


def grid_search_svr(grid=None, data_dict: dict = FULL_DATA_DICT, pp_dict: dict = PP_DICT):
    param_grid = grid or [
        # {
        #     'kernel': ('poly',),
        #     'C': (100, 1, 10),
        #     'epsilon': (0.01, 0.1, 0.2, 0.5),
        #     'shrinking': (True,),
        #     'tol': (1e-3,),
        #     'degree': (3, 9, 15)
        # },
        {
            'kernel': ('rbf',),
            'C': (100, 1, 10),
            'epsilon': (0.01, 0.1, 0.2, 0.5),
            'shrinking': (True,),
            'tol': (1e-3,)
        }
    ]
    gscv.perform_grid_search(SVR(), param_grid, data_dict, pp_dict, f'gs_svr.log')


def grid_search_rf(grid=None, data_dict: dict = FULL_DATA_DICT, pp_dict: dict = PP_DICT):
    param_grid = grid or [{
        'n_estimators': (10, 25, 200),
        'criterion': ('mse',),
        'max_depth': (10, 25),
        'max_features': ('auto',),
        'ccp_alpha': (0.0, 0.1,),
        'max_samples': (0.85, None)
    }]
    gscv.perform_grid_search(RandomForestRegressor(), param_grid, data_dict, pp_dict, f'gs_rf.log')


if __name__ == '__main__':
    # grid_search_knn()
    # grid_search_svr()
    # grid_search_rf()
    pass
