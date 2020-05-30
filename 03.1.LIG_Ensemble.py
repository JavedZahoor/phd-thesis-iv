import os
path = '/Users/Javed Zahoor/Downloads/PhD/';
os.chdir(path)
from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader

import numpy
import time
from time import sleep
import scipy.io
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
#from sklearn.model_selection import cross_validation
from sklearn.model_selection import LeaveOneOut
#https://stackoverflow.com/questions/44309544/sklearn-gridsearchcv-typeerror-leaveoneout-object-is-not-iterable
from sklearn import metrics

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import matthews_corrcoef,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import LeaveOneOut
#import impute.SimpleImputer

from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
#import sklearn.cross_validation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


A_LIGs=['datasetA_trained_clfs/Standard-LOOCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-DT-JMIM-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-ExtraTree-JMIM-50.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-ExtraTree-JMIM-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-DT-JMIM-150.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-DT-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-ExtraTree-JMI-100.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-DT-JMI-250.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-ExtraTree-JMIM-10.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-DT-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-DT-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-DT-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-DT-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-ExtraTree-JMI-100.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-DT-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-ExtraTree-JMIM-50.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-250.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-ExtraTree-JMIM-100.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-DT-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-DT-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-ExtraTree-JMIM-10.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-DT-JMI-250.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-DT-MRMR-100.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-ExtraTree-JMI-50.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-ExtraTree-JMIM-100.joblib.pkl, ' ,'datasetA_trained_clfs/Quantile-LOOCV-ExtraTree-JMIM-100.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-DT-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-ExtraTree-MRMR-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-ExtraTree-MRMR-250.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-ExtraTree-JMIM-100.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-DT-JMIM-150.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-DT-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-DT-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-ExtraTree-JMI-100.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-ExtraTree-JMIM-10.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-DT-JMIM-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-DT-JMI-200.joblib.pkl, ' ,'datasetA_trained_clfs/Standard-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Quantile-10FoldCV-DT-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-DT-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Robust-LOOCV-DT-JMIM-250.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-DT-JMIM-150.joblib.pkl' ,'datasetA_trained_clfs/Robust-10FoldCV-ExtraTree-MRMR-200.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-ExtraTree-MRMR-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-DT-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-100.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-ExtraTree-JMI-100.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-DT-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-AdaBoost-JMIM-10.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-AdaBoost-JMIM-10.joblib.pkl' ,'datasetA_trained_clfs/Standard-10FoldCV-DT-JMIM-150.joblib.pkl' ,'datasetA_trained_clfs/Imputer-10FoldCV-DT-JMIM-200.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Standard-LOOCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Quantile-LOOCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetA_trained_clfs/Imputer-LOOCV-AdaBoost-JMI-150.joblib.pkl']
A_LIG_Accuracies = [0.7583635061, 0.7248059545, 0.7120943041, 0.7088976351, 0.7066444632, 0.7066444632, 0.7066444632, 0.7028057401, 0.6981908753, 0.6975322047, 0.6973248698, 0.6964608344, 0.6944735981, 0.6931258766, 0.6906524788, 0.6906122449, 0.6896299473, 0.6852710843, 0.6852710843, 0.6852710843, 0.6852710843, 0.6850174626, 0.6848189133, 0.6847219889, 0.6847219889, 0.6847219889, 0.6847219889, 0.6841677968, 0.6840240338, 0.6840240338, 0.6840240338, 0.6836195854, 0.6830747026, 0.6781374717, 0.6778231293, 0.6778231293, 0.6752146334, 0.6751871149, 0.6751871149, 0.6751871149, 0.6751871149, 0.6751194533, 0.6735600907, 0.6719614512, 0.6715872332, 0.6715872332, 0.6715872332, 0.6684780331, 0.6658062357, 0.6654324053, 0.6650340136, 0.6650340136, 0.6650340136, 0.6650340136, 0.6644708815, 0.6643324329, 0.6634353741, 0.6634353741, 0.6630038093, 0.6623972492, 0.661903597, 0.6608867054, 0.6607901795, 0.6555230324, 0.6537306073, 0.6534502323, 0.6533336342, 0.6533336342, 0.652244898, 0.652244898, 0.652244898, 0.6518233257, 0.6516925953, 0.6502040816, 0.6501133787, 0.6485147392, 0.6485147392, 0.6482924797, 0.6479655952, 0.646713632, 0.646713632, 0.646195432, 0.6437545162, 0.6417027715, 0.6412348632, 0.6412348632, 0.6412348632, 0.6405215419, 0.6403010407, 0.6397011269, 0.6395637623, 0.6395637623, 0.6394588316, 0.6394588316, 0.6394557823, 0.6394557823, 0.6394557823, 0.6394557823]

B_LIGs=['datasetB_trained_clfs/Standard-LOOCV-ExtraTree-JMI-100.joblib.pkl' ,'datasetB_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-100.joblib.pkl' ,'datasetB_trained_clfs/Robust-LOOCV-ExtraTree-JMIM-250.joblib.pkl' ,'datasetB_trained_clfs/Standard-LOOCV-ExtraTree-JMIM-50.joblib.pkl' ,'datasetB_trained_clfs/Robust-10FoldCV-ExtraTree-JMIM-250.joblib.pkl' ,'datasetB_trained_clfs/Standard-10FoldCV-ExtraTree-JMIM-50.joblib.pkl' , 'datasetB_trained_clfs/Standard-LOOCV-ExtraTree-JMI-10.joblib.pkl']
B_LIG_Accuracies = [0.1536772991,0.1338590549,0.1221636963,0.1197286504,0.1188721579,0.110793335,0.1034369376]

C_LIGs=['datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-JMIM-150.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-JMIM-150.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-JMIM-150.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-JMIM-150.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-MRMR-250.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-JMI-250.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-JMIM-250.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-50.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-50.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-MRMR-50.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-MRMR-50.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-ExtraTree-MRMR-50.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-ExtraTree-MRMR-50.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-100.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-ExtraTree-MRMR-100.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-MRMR-200.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-JMI-200.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-ExtraTree-JMIM-200.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-MRMR-150.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-JMI-150.joblib.pkl' ,'datasetC_trained_clfs/Standard-10FoldCV-AdaBoost-JMIM-150.joblib.pkl' ,'datasetC_trained_clfs/Robust-10FoldCV-AdaBoost-JMIM-150.joblib.pkl' ,'datasetC_trained_clfs/Quantile-10FoldCV-AdaBoost-JMIM-150.joblib.pkl' ,'datasetC_trained_clfs/Imputer-10FoldCV-AdaBoost-JMIM-150.joblib.pkl']
C_LIG_Accuracies = [0.7447736872, 0.7447736872, 0.7447736872, 0.7447736872, 0.7447736872, 0.7447736872, 0.7447736872, 0.7447736872, 0.7447736872, 0.7229216542, 0.7229216542, 0.7229216542, 0.682207187, 0.682207187, 0.682207187, 0.6812429872, 0.6812429872, 0.6812429872, 0.6812429872, 0.6812429872, 0.6812429872, 0.6812429872, 0.6812429872, 0.6812429872, 0.6810702937, 0.6810702937, 0.6810702937, 0.6810702937, 0.6810702937, 0.6810702937, 0.6810702937, 0.6810702937, 0.6810702937, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6648959187, 0.6536860754, 0.6536860754, 0.6536860754, 0.6500634512, 0.6500634512, 0.6500634512, 0.6500634512, 0.6500634512, 0.6500634512, 0.6500634512, 0.6500634512, 0.6500634512, 0.6490338189, 0.6490338189, 0.6490338189, 0.6490338189, 0.6490338189, 0.6490338189, 0.6459092906, 0.6459092906, 0.6384963826, 0.6384963826, 0.6384963826, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868, 0.6319504868]

basePath = '/Users/Javed Zahoor/Downloads/PhD/';
'''
Load each of the trained LIGs from datasetB_trained_clfs
Figure out which variant of the dataset from dataset<DataSet>_pickles needs to be loaded and load it.
    Note 1: for dataset A the pickles need to be generated from selected_features\datasetA first.
    Note 2: This Should be done for validate dataset for all the three datasets and validation dataset should be used??
    Note 3: Or the validation set should be used as is and just passed on to the trained classifiers to predict
    split lig on / to remove the folder name
    split the right side on - to make parts of it
    parts[0] = PreProcessing Method
    parts[1] = Validation method
    parts[2] = classifier
    parts[3] = FSS technique
    parts[4] = FSS Size
    then the dataset pickle is
    'dataset' + Dataset + '_train' + parts[4] + '-' + parts[3] +'.joblib'
Once the dataset has been loaded (and decided in the previous step)
Use the corresponding LIG to predict classes for the loaded validation set
    For each of the validation instance from prediction
        Store the instance ID, the classifier ID, the actual class
        For each of the classifier the predicted class score and the class label in two separate files/matrices
stored the matrics
load the matrics in excel and check if majority voting OR simple sum or weighted avg of acc / mcc / prod works better

Store the ensemble outputs to basePath + "\Infiltration_ensembles\Dataset.lig.csv"
'''
Datasets = ["C"] #"B","A",
#Dataset ="B" #for testing purposes
for Dataset in Datasets:
    if (eval("len(" + Dataset + "_LIG_Accuracies) != len(" + Dataset + "_LIGs)")):
        print Dataset + "_LIG mismatches the accuracies list";
    actuals = "";
    results = "";
    LIGs = eval(Dataset + "_LIGs");
    LIG_Accuracies = eval(Dataset + "_LIG_Accuracies")
    start_time = time.time();
    padding = 0;
    #load the dataset
    d = DataSetLoader();
    #X_train_full = d.LoadDataSet(Dataset+"_train");
    #y_train = d.LoadDataSetClasses(Dataset+"_train");
    #targets=list(numpy.transpose(y_train))
    #y_train=[]
    #for i in targets:
    #    y_train.append(int(i))


    X_validate_full = d.LoadDataSet(Dataset + "_test");
    y_validate = d.LoadDataSetClasses(Dataset + "_test");
    print ("Dimensions of validation data and labels:", X_validate_full.shape, y_validate.shape)
    targets = list(numpy.transpose(y_validate))
    y_validate = []
    if Dataset == "C":
        y_validate = numpy.array(targets)
        #y_validate[y_validate == 0] = -1
    else:
        for i in targets:
            y_validate.append(int(i))

    y_test = y_validate

    actuals = ','.join([str(elem) for elem in y_test])
    actuals = actuals.replace("\n","").replace("[","").replace("]","").replace(" ",",") #to handle dataset C targets
    print("actuals=");
    print(actuals);
    #results[0][75]
    predictions = ""
    for lig in LIGs:
        #lig="datasetB_trained_clfs/Standard-LOOCV-ExtraTree-JMI-100.joblib.pkl" #for debugging only
        #load LIG member name
        parts = lig.split("/")[1].split(".")[0].split("-")

        #extract clf specific indices file name
        datasetPickle = 'dataset' + Dataset + '_train' + parts[4] + '-' + parts[3] +'.joblib'

        #load clf specific indices
        clf_specific_indicies = joblib.load(basePath + 'selected_features/dataset' + Dataset + "/" + datasetPickle + ".pkl");


        #choose the subset
        #X_train = X_train_full[:, clf_specific_indicies]
        X_validate = X_validate_full[:, clf_specific_indicies]
        X_test = X_validate

        #load the classifier
        clf = joblib.load(basePath + lig);

        #retrieve accuracy from the existing records instead of trying to recalculate it.

        #predict classes
        pred = clf.predict(X_validate)
        #load CV accuracy of the clf on training datasets
        if parts[1]=="LOOCV":
            cv_size = len(pred)/2;
        else:
            cv_size = 10;
        predictions = ','.join([str(elem) for elem in pred])
        predictions = str(LIG_Accuracies[LIGs.index(lig)]) + "," + predictions

        #Append it to an overall results string

        results = results + predictions + "\n"
        #cross_val_score(clf, X_train, y_train, cv=cv_size, n_jobs=-1) #?? do we need this?
        #accuracy = accuracy_score(y_test, pred)#?? do we need this?
        #mcc = matthews_corrcoef(y_test, pred) #?? do we need this?
        #TODO: END OF LIG LOOP

    #TODO: keep this outside the ligs loop but inside the Databases loop
    #Open the file, Dump the results in the file and close it
    f=open(basePath + 'Infiltration_ensembles/lig'+Dataset+'-Ensemble.txt','a');
    f.write(Dataset + "," + actuals + "\n") #str(LIGs[LIGs.index(lig)]) -> Clf name
    f.write(results)
    f.close()
    # TODO: END OF Datasets LOOP
end_time = time.time();
timeTaken = end_time - start_time
#roundedResults = eval("[[0] * len(LIGs)] * " + str(len(pred) + padding))
#import numpy as np

#roundedResults = np.zeros((len(pred)+2,len(LIGs)))

#store the classes in the matrix to dump into the file
#roundedResults [padding:len(pred), LIGs.index(lig)] = pred
# place the predictor values in the matrix roundedResults[:][0] = pred
# extract roundedResults[22][0] 22th element of 0th classifier