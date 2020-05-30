import numpy as np
from sklearn.externals import joblib
import scipy.io as io
train_dataset = io.loadmat('datasetC_raw/training_dataset.mat')
X=train_dataset['training_dataset']
train_classes = io.loadmat('datasetC_raw/train_classes.mat')
y=train_classes['train_classes']
joblib.dump(X, 'datasetC_raw/DatasetC_Train.joblib.pkl')
joblib.dump(y, 'datasetC_raw/DatasetC_TrainClasses.joblib.pkl')

validate_dataset = io.loadmat('datasetC_raw/validation_dataset.mat')
X=validate_dataset['validation_dataset']
validate_classes = io.loadmat('datasetC_raw/validate_classes.mat')
y=validate_classes['validate_classes']
joblib.dump(X, 'datasetC_raw/DatasetC_Validation.joblib.pkl')
joblib.dump(y, 'datasetC_raw/DatasetC_ValidationClasses.joblib.pkl')