from sklearn.metrics.regression import mean_absolute_error
from sklearn.svm import SVR

from src.model import cfw

X_train, X_test, y_train, y_test = cfw.load_train_test_split(dataset='CNN', feature_frame=True)

# svr = SVR(degree=10, kernel='poly', C=1, epsilon=0.05)

# Using optimal params from Grid Search
svr = SVR(degree=10, kernel='linear', C=2, epsilon=0.05)
X_train, X_test, y_train, y_test = cfw.preprocess_data(X_train, X_test,
                                                       y_train, y_test,
                                                       n_components=0.95)

y_train = y_train.iloc[:, 0].values
y_test = y_test.iloc[:, 0].values

svr.fit(X_train, y_train.ravel())

y_pred = svr.predict(X_test)

m_err = mean_absolute_error(y_test.ravel(), y_pred)
