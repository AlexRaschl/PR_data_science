import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from tqdm import tqdm

from src.config import BL_DATA_DICT
from src.model.utils import train_test_model, handle_savefig

# TODO GS PARAMS
params = {}
pp_dict = {'n_components': -1, 'std_scale': True, 'with_mean': True, 'log_tf': True}
data_dict = BL_DATA_DICT

models = [KNeighborsRegressor(), SVR(), RandomForestRegressor(), MLPRegressor()]
names = [type(model).__name__ for model in models]
fitted = []
errors = []
r2s = []

for model in tqdm(models):
    pipeline, m_err, r2 = train_test_model(model,
                                           params,
                                           data_dict,
                                           pp_dict,
                                           save_model=True)
    fitted.append(pipeline)
    errors.append(m_err)
    r2s.append(r2)

fig = plt.figure(figsize=(10, 5))
sns.barplot(x=names[:-1], y=errors[:-1])
handle_savefig(['basic_models', 'error_comparison'])
