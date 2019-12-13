from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.model import cfw

X_train, X_test, y_train, y_test = cfw.load_train_test_split()

# svr = SVR(degree=10, kernel='poly', C=10)
svr = KNeighborsRegressor(n_neighbors=7)

X_train = cfw.feature_data_frame(X_train)
X_test = cfw.feature_data_frame(X_test)

X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

y_train = y_train.iloc[:, 0].values
y_test = y_test.iloc[:, 0].values
# y_scaler = StandardScaler().fit(y_train)

# y_train = y_scaler.transform(y_train)
# y_test = y_scaler.transform(y_test)


svr.fit(X_train, y_train.ravel())

y_pred = svr.predict(X_test)

m_err = mean_absolute_error(y_test.ravel(), y_pred)
