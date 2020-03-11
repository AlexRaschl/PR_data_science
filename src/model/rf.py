from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.regression import mean_absolute_error

from src.config import N_JOBS
from src.model import cfw

X_train, X_test, y_train, y_test = cfw.load_train_test_split(dataset='CNN',
                                                             feature_frame=True,
                                                             n_labels=5)

# svr = SVR(degree=10, kernel='poly', C=1, epsilon=0.05)

# Using optimal params from Grid Search
rf = RandomForestRegressor(n_jobs=N_JOBS)
X_train, X_test, y_train, y_test = cfw.preprocess_data(X_train, X_test,
                                                       y_train, y_test,
                                                       n_components=0.75,
                                                       std_scale=False)

y_train = y_train.iloc[:, 0].values
y_test = y_test.iloc[:, 0].values

rf.fit(X_train, y_train.ravel())

y_pred = rf.predict(X_test)

m_err = mean_absolute_error(y_test.ravel(), y_pred)

y_pred.min()
