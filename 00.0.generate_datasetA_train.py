import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit

XX = joblib.load('datasetA_raw/DatasetA_Validation.joblib.pkl')
X = np.array(XX)
yy = joblib.load('datasetA_raw/DatasetA_ValidationClasses.joblib.pkl')
y = np.array(yy)

sss = StratifiedShuffleSplit(n_splits=1, train_size=70, random_state=0)
for train_index, test_index in sss.split(X, y):
    train_i = train_index
    test_i = test_index


X_train, X_test = X[train_i], X[test_i]
y_train, y_test = y[train_i], y[test_i]

#dump X_train and X_test as pickles
joblib.dump(X_train, 'datasetA_raw/DatasetA_Train.joblib.pkl')
joblib.dump(y_train, 'datasetA_raw/DatasetA_TrainClasses.joblib.pkl')