import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model import cfw

X_train, X_test, y_train, y_test = cfw.load_train_test_split(feature_frame=False, one_hot_inputs=False, n_labels=5)
# cls = cfw.get_textual_description(X_test,  n_labels=1)


X_train, X_test, y_train, y_test = cfw.load_train_test_split(feature_frame=True, one_hot_inputs=True, n_labels=5)

mlp = make_pipeline(StandardScaler(with_mean=False),
                    MLPRegressor(hidden_layer_sizes=(100, 50, 50, 25, 10), solver='lbfgs', verbose=True,
                                 max_iter=1000))

mlp.fit(X_train, y_train.iloc[:, 0].ravel())

y_pred = mlp.predict(X_test)
# y_pred = np.array([y_train.median() for i in range(len(y_pred))])


print(f'Median of predictions: {np.median(y_pred)}')

m_err = mean_absolute_error(y_test.iloc[:, 0].ravel(), y_pred.ravel())

print(f'Mean absolute error: {m_err}')
r2 = mlp.score(X_test, y_test.iloc[:, 0].ravel())
print(f'R^2={r2}')
