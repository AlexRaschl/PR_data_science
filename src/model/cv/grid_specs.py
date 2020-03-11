from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.model.cv import cross_validation as gscv


def grid_search_knn(datasets, grid=None):
    param_grid = grid or [
        {
            'n_neighbors': range(1, 10, 5),
        }
    ]
    for ds in datasets:
        gscv.perform_grid_search(KNeighborsRegressor(), param_grid, f'gs_knn_{ds}.log',
                                 dataset=ds, one_hot_inputs=True, n_labels=2)


def grid_search_svr(datasets, grid=None):
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
        gscv.perform_grid_search(RandomForestRegressor, param_grid, f'gs_svr_{ds}.log',
                                 dataset=ds,
                                 )


def grid_search_rf(datasets, grid=None):
    param_grid = grid or [
        {
            'n_estimators': (10, 15, 100, 150),
            'criterion': ('mse', 'mae'),
            'max_depth': (None, 15, 25),
            'max_features': ('sqrt', None),
            'ccp_alpha': (0.0, 0.1),
            'max_samples': (None, 0.75)
        }
    ]
    for ds in datasets:
        gscv.perform_grid_search(RandomForestRegressor(), param_grid, f'gs_rf_{ds}.log',
                                 dataset=ds,
                                 )


grid_search_rf(['CNN'])
