from sklearn.ensemble import RandomForestRegressor

from src.config import N_JOBS, PP_DICT, FULL_DATA_DICT
from src.model.utils import train_test_model

# Done: Best Grid Search params
params = {
    'n_jobs': N_JOBS,
    'criterion': 'mse',
    'max_depth': 25,
    'max_features': 'auto',
    'n_estimators': 200,
    'max_samples': 0.75,
    'ccp_alpha': 0.0
}
pp_dict = PP_DICT
data_dict = FULL_DATA_DICT

pipeline, m_err, r2 = train_test_model(RandomForestRegressor(), params, data_dict, pp_dict, save_model=True)
