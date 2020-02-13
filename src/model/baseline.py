import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.regression import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.config import VIS_PATH
from src.model import cfw

X_train, X_test, y_train, y_test = cfw.load_train_test_split()

knn = KNeighborsRegressor(n_neighbors=1)

X_train = cfw.feature_data_frame(X_train)
X_test = cfw.feature_data_frame(X_test)

X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

# y_scaler = StandardScaler().fit(y_train)
# y_train = y_scaler.transform(y_train)
# y_test = y_scaler.transform(y_test)

knn.fit(X_train, y_train.iloc[:, 0].ravel())

y_pred = knn.predict(X_test)
print(np.median(y_pred))

m_err = mean_absolute_error(y_test.iloc[:, 0].ravel(), y_pred.ravel())
# print(y_scaler.inverse_transform([m_err]))
print(m_err)
r2 = knn.score(X_test, y_test.iloc[:, 0].ravel())
print(f'R^2={r2}')

fig, ax = plt.subplots(1, 4, figsize=(20, 5),
                       gridspec_kw={'top': 0.9, 'left': 0.1, 'right': 0.92, 'height_ratios': [1]})
plt.suptitle(f'Test/Pred comparison: R^2={r2}, m_abs={m_err}')
ax[0].hist(y_test.iloc[:, 0].ravel(), bins=25, color='red')
ax[0].set_title('Test set distribution')
ax[1].hist(y_pred.ravel(), bins=25)
ax[1].set_title('Prediction distribution')
ax[2].boxplot(y_test.iloc[:, 0].ravel(), boxprops=dict(color='red'))
ax[2].set_title('Test set summary statistics')
ax[2].set_xticks([])
ax[3].boxplot(y_pred.ravel())
ax[3].set_title('Predictions summary statistics')
ax[3].set_xticks([])

# plt.show()
plt.savefig(VIS_PATH + 'baseline_test_pred_comparison.png')
