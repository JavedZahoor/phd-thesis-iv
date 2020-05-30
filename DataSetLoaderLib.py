from GlobalUtils import *
import scipy
import os
from MachineSpecificSettings import Settings
from sklearn.externals import joblib
import pickle;

#TODO: use this if needed os.getcwd()+"/"+
class DataSetLoader(object):
    @timing
    def LoadDataSet(self, dataSetType):
        s = Settings();
        if dataSetType == "A_train":
		variables=numpy.array(joblib.load('datasetA_raw/DatasetA_Train_mean.joblib.pkl'))
		return variables;
	elif dataSetType == "A_test":
		variables=numpy.array(joblib.load('datasetA_raw/DatasetA_Validation_mean.joblib.pkl'))
		return variables;
	elif dataSetType == "B_train":
		variables=numpy.array(joblib.load('datasetB_raw/DatasetB_Train.joblib.pkl'))
		return variables;
	elif dataSetType == "B_test":
		variables=numpy.array(joblib.load('datasetB_raw/DatasetB_Validation.joblib.pkl'))
		return variables;
        elif dataSetType == "C_train":
		variables=numpy.array(joblib.load('datasetC_raw/DatasetC_Train.joblib.pkl'))
		return variables;
	elif dataSetType == "C_test":
		variables=numpy.array(joblib.load('datasetC_raw/DatasetC_Validation.joblib.pkl'))
		return variables;

        else:
		print "INVALID INPUT"
        	logWarning("HARD CODED VALUE from DataSetLoaderLib.LoadDataSet()");
        	return [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]];
    @timing
    def LoadDataSetClasses(self, dataSetType):
        s = Settings();
        if dataSetType == "A_train":
		variables=numpy.array(joblib.load('datasetA_raw/DatasetA_TrainClasses.joblib.pkl'))
		return variables;
	elif dataSetType == "A_test":
		variables=numpy.array(joblib.load('datasetA_raw/DatasetA_ValidationClasses.joblib.pkl'))
		return variables;
	elif dataSetType == "B_train":
		variables=numpy.array(joblib.load('datasetB_raw/DatasetB_TrainClasses.joblib.pkl'))
		return variables;
	elif dataSetType == "B_test":
		variables=numpy.array(joblib.load('datasetB_raw/DatasetB_ValidationClasses.joblib.pkl'))
		return variables;
	elif dataSetType == "C_train":
		variables=numpy.array(joblib.load('datasetC_raw/DatasetC_TrainClasses.joblib.pkl'))
		return variables;
	elif dataSetType == "C_test":
		variables=numpy.array(joblib.load('datasetC_raw/DatasetC_ValidationClasses.joblib.pkl'))
		return variables;

        else:
		print "INVALID INPUT"
        	logWarning("HARD CODED VALUE from DataSetLoaderLib.LoadDataSetClasses()");
        	return [0, 1, 1, 1, 0, 1];