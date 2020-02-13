from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.model.cv import cross_validation as gscv


def grid_search_knn(datasets, grid=None, n_components=0.95):
    param_grid = grid or [
        {
            'n_neighbors': range(1, 100, 5),
        }
    ]
    for ds in datasets:
        gscv.perform_grid_search(KNeighborsRegressor(), param_grid, f'gs_knn_{ds}.log',
                                 n_components=n_components)


def grid_search_svr(datasets, grid=None, n_components=0.95):
    param_grid = grid or [
        {
            'kernel': ['rbf', 'poly', 'linear'],
            'C': (0.2, 0.5, 0.7, 0.9, 1, 1.1, 1.2, 1.5, 2),
            'epsilon': (0.01, 0.05, 0.1, 0.2, 0.5),
            'shrinking': (True, False),
            'tol': (1e-3, 1e-4, 1e-2)
        }
    ]
    for ds in datasets:
        gscv.perform_grid_search(SVR(), param_grid, f'gs_svr_{ds}.log',
                                 n_components=n_components)


grid_search_svr('CNN')
