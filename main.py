import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('svm-data.csv', header=None)
X = data.as_matrix(columns=[data.columns[1], data.columns[2]])
y = data.as_matrix(columns=[data.columns[0]])

clf = SVC(kernel='linear', random_state=241, C=100000)
clf.fit(X, y)
