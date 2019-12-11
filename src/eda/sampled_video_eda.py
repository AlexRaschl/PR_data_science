import matplotlib.pyplot as plt

from src.preprocessing.indexer import Indexer

X_train, X_test, y_train, y_test = Indexer.load_split('cache/tts_42')

# Merge dataframes
df = X_train.append(X_test)
y = y_train.append(y_test)

df['viewcounts'] = y

df.describe()
df.info()

df.head()

samples = df.sort_values(by='viewcounts')

df.hist(column=['viewcounts'], bins=100)
plt.show()

y_train.hist(bins=100)
plt.show()

y_test.hist(bins=100)
plt.show()
