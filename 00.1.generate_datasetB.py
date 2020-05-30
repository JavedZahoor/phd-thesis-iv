import numpy as np
from sklearn.externals import joblib
import scipy.io as io
mat = io.loadmat('datasetB_raw/DataSetBGSE24417MAQCIITraining.mat')
X=mat['data']
y=mat['target']
joblib.dump(X, 'datasetB_raw/DatasetB_Train.joblib.pkl')
joblib.dump(y, 'datasetB_raw/DatasetB_TrainClasses.joblib.pkl')

mat = io.loadmat('datasetB_raw/DataSetBGSE24417MAQCIIValidation.mat')
X=mat['data']
y=mat['target']
joblib.dump(X, 'datasetB_raw/DatasetB_Validation.joblib.pkl')
joblib.dump(y, 'datasetB_raw/DatasetB_ValidationClasses.joblib.pkl')