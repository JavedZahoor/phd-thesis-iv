import os
path = '/Users/Javed Zahoor/Downloads/PhD/'; #'/home/ubuntu/jz/'
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


A_FTs=['datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMI-250.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMI-250.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-MRMR-10.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-MRMR-50.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-MRMR-100.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-MRMR-150.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-MRMR-200.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-MRMR-250.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMI-10.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMI-50.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMI-100.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMI-150.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMI-200.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMI-250.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMIM-10.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMIM-50.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMIM-100.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMIM-150.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMIM-200.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-SVM-JMIM-250.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-MRMR-10.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-MRMR-50.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-MRMR-100.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-MRMR-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-MRMR-200.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-MRMR-250.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMI-10.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMI-50.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMI-100.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMI-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMI-200.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMI-250.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMIM-10.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMIM-50.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMIM-100.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMIM-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMIM-200.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-SVM-JMIM-250.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMI-200.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMIM-10.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMI-10.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMI-250.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-MRMR-150.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-JMI-250.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMI-100.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Imputer-LOOCV-RandomForest-JMIM-150.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMI-50.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-MRMR-100.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-MRMR-100.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMI-250.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-MRMR-150.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMI-100.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMIM-150.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-50.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-10.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMIM-200.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMI-200.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMIM-200.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-JMI-50.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMI-150.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-MRMR-150.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMI-150.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-JMIM-250.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-JMIM-200.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetA_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Robust-LOOCV-RandomForest-JMI-50.joblib.pkl','datasetA_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-JMI-200.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMIM-50.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-SVM-JMIM-100.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-RandomForest-JMI-10.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-100.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-JMI-100.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-SVM-JMIM-10.joblib.pkl','datasetA_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-10.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-SVM-JMI-100.joblib.pkl','datasetA_trained_clfs/Standard-LOOCV-SVM-JMIM-200.joblib.pkl','datasetA_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-10.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-MRMR-250.joblib.pkl','datasetA_trained_clfs/Quantile-LOOCV-SVM-JMI-150.joblib.pkl']
A_FT_Accuracies = [0.6879478458,0.662244898,0.6612886859,0.6548871669,0.6531070615,0.6527574525,0.649804826,0.646713632,0.6447710237,0.6405483635,0.6405483635,0.6402476823,0.6394557823,0.6389143092,0.6389143092,0.6351927438,0.635187309,0.6306965251,0.6277333088,0.6266666667,0.6229414514,0.6200657126,0.6200286264,0.6174451738,0.6174451738,0.6170373212,0.6162698126,0.6146889991,0.613877551,0.6133577368,0.6133325891,0.6128117914,0.6127327119,0.6120748299,0.6117315147,0.6100680272,0.6096180936,0.6081561797,0.6076280539,0.6068480408,0.6056093618,0.6056093618,0.6020408163,0.6010884354,0.5970999662,0.5969664295,0.5964608685,0.5960633343,0.5957645461,0.5943754294,0.5939233299,0.5928397792,0.5928397792,0.5916079783,0.5888660216,0.5887039781,0.5878011644,0.5878011644,0.5874914966,0.5866840693,0.5859149763,0.5856349206,0.5848995258,0.5845296289,0.5833435993,0.5828269088,0.5823166941,0.581828327,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5805404821,0.5797614413,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5795343461,0.5783112286,0.5766896112,0.5750228782,0.5712123499,0.571112956,0.5706703602,0.570632219,0.5704604556,0.5691212862,0.5688762647,0.5682440406,0.5681743289,0.5651012407,0.5648522348,0.5646752443,0.5646258503,0.5621722512,0.5619686037,0.5606704554,0.5600691064,0.5582950778,0.5581311511,0.5580257638,0.5565434662,0.555379633,0.5549258915,0.5542743764,0.5535994971,0.5517903484,0.5512719333,0.5512719333,0.5512719333,0.5508594696,0.5508011841,0.5502754059,0.5493968587,0.5493968587,0.5474573898,0.546585869,0.5465034187,0.5456251912,0.5444898363,0.5444661089,0.5440809745,0.542744605,0.542744605,0.542014977,0.5415702948,0.5415183302,0.5395603436,0.5372185019,0.5367372623,0.5356955525,0.535060485,0.5348756864,0.5343760064,0.5343487499,0.533305039,0.5331562529,0.533146298,0.5313376933,0.5297959184,0.529628277,0.5290618203,0.528771038,0.5278335007,0.5267715419,0.5238466125,0.5227547368,0.5216953223,0.5210348208,0.5170786627,0.5149662087,0.5134029497,0.5130516557,0.5127269145,0.5107219358,0.5094930527,0.5083542251,0.5081632653,0.5075464013,0.506122449,0.5060913686,0.5059057954,0.5038757973,0.5035809013,0.5033533601,0.4997506511,0.497486068,0.4963510375,0.4955477014,0.4954884638,0.4929319829,0.4928668318]
#TODO: these are actually indexes i.e. accuracy x mcc and not accuracies

B_FTs=['datasetB_trained_clfs/Robust-LOOCV-RandomForest-MRMR-250.joblib.pkl','datasetB_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetB_trained_clfs/Robust-LOOCV-SVM-JMI-50.joblib.pkl','datasetB_trained_clfs/Standard-LOOCV-RandomForest-JMIM-100.joblib.pkl','datasetB_trained_clfs/Robust-LOOCV-SVM-JMI-100.joblib.pkl','datasetB_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-100.joblib.pkl','datasetB_trained_clfs/Imputer-LOOCV-RandomForest-MRMR-150.joblib.pkl','datasetB_trained_clfs/Robust-LOOCV-RandomForest-MRMR-150.joblib.pkl','datasetB_trained_clfs/Robust-10FoldCV-SVM-JMI-50.joblib.pkl']
B_FT_Accuracies = [0.1447541011,0.1349675267,0.1194717068,0.1182401934,0.1099504452,0.1048572825,0.1035671821,0.1032875666,0.1018954787]

C_FTs=['datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-MRMR-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-JMI-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-JMIM-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-MRMR-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-JMI-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-JMIM-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-MRMR-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-JMI-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-JMIM-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-MRMR-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-JMI-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-JMIM-250.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-MRMR-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-JMI-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-JMIM-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-MRMR-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-JMI-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-JMIM-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-MRMR-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-JMI-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-JMIM-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-150.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-JMI-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-MRMR-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-JMI-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-JMIM-150.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-MRMR-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-JMI-250.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-JMIM-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-MRMR-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-JMI-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-JMIM-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-150.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-MRMR-150.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-JMI-150.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-JMIM-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-MRMR-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-JMI-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-JMIM-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-MRMR-100.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-MRMR-100.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-JMI-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-RandomForest-JMIM-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-JMIM-250.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-JMI-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-MRMR-100.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-MRMR-100.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-MRMR-50.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-MRMR-50.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-MRMR-50.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-MLP-MRMR-50.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-MRMR-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-JMI-150.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-JMIM-150.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-MRMR-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-JMI-200.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-JMIM-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-RandomForest-MRMR-100.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-MRMR-50.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-SVM-MRMR-50.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-MRMR-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-JMI-150.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-SVM-JMIM-150.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-MRMR-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-JMI-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-JMIM-250.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-MRMR-50.joblib.pkl','datasetC_trained_clfs/Robust-10FoldCV-SVM-MRMR-50.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-50.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-MRMR-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-JMI-200.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-MLP-JMIM-200.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-MRMR-50.joblib.pkl','datasetC_trained_clfs/Quantile-10FoldCV-MLP-MRMR-50.joblib.pkl','datasetC_trained_clfs/Standard-10FoldCV-RandomForest-MRMR-50.joblib.pkl','datasetC_trained_clfs/Imputer-10FoldCV-RandomForest-MRMR-50.joblib.pkl']
C_FT_Accuracies = [0.7587805845,0.7587410783,0.7574383685,0.7559718636,0.7559718636,0.7559718636,0.7557011527,0.7549899747,0.7542151929,0.7542151929,0.7542151929,0.7534504185,0.7534504185,0.7534504185,0.7508637608,0.7499649385,0.7481851745,0.7481851745,0.7481851745,0.7469518957,0.7464377578,0.7451261844,0.7447736872,0.7447736872,0.7433794161,0.7432186545,0.7432186545,0.7432186545,0.7419207454,0.7419207454,0.7412487153,0.7410724667,0.7410724667,0.7410724667,0.7408785933,0.7407023447,0.7400880072,0.7400880072,0.7387869795,0.7387869795,0.7387869795,0.7386526949,0.7386526949,0.7386526949,0.7373712462,0.7373712462,0.7370011242,0.7349442021,0.7339467061,0.7339467061,0.7339467061,0.7334937771,0.732560019,0.732560019,0.732560019,0.732149755,0.731118813,0.7299329214,0.7299329214,0.7299329214,0.7274673051,0.7271766271,0.7271766271,0.7271766271,0.7258274676,0.7258274676,0.7258274676,0.7240773894,0.7240773894,0.7234680346,0.7234680346,0.7234680346,0.7220801812,0.7211968719,0.7211968719,0.7186618265,0.7164078528,0.7164078528,0.7161840785,0.7161840785,0.7131223382,0.7131223382,0.7118163516,0.7118163516,0.7118163516,0.7089916391,0.7089916391,0.7089916391,0.7087153048,0.706051086,0.7054385731,0.7054385731,0.6992439231,0.6992439231,0.6992439231,0.697706597,0.697706597,0.697706597,0.6817705141,0.6817705141,0.6787003647,0.6775475214,0.6775475214,0.6775475214,0.6728361335,0.6728361335,0.6671439677,0.6671272372]

basePath = '/Users/Javed Zahoor/Downloads/PhD/';
'''
Load each of the trained FTs from datasetB_trained_clfs
Figure out which variant of the dataset from dataset<DataSet>_pickles needs to be loaded and load it.
    Note 1: for dataset A the pickles need to be generated from selected_features\datasetA first.
    Note 2: This Should be done for validate dataset for all the three datasets and validation dataset should be used??
    Note 3: Or the validation set should be used as is and just passed on to the trained classifiers to predict
    split ft on / to remove the folder name
    split the right side on - to make parts of it
    parts[0] = PreProcessing Method
    parts[1] = Validation method
    parts[2] = classifier
    parts[3] = FSS technique
    parts[4] = FSS Size
    then the dataset pickle is
    'dataset' + Dataset + '_train' + parts[4] + '-' + parts[3] +'.joblib'
Once the dataset has been loaded (and decided in the previous step)
Use the corresponding FT to predict classes for the loaded validation set
    For each of the validation instance from prediction
        Store the instance ID, the classifier ID, the actual class
        For each of the classifier the predicted class score and the class label in two separate files/matrices
stored the matrics
load the matrics in excel and check if majority voting OR simple sum or weighted avg of acc / mcc / prod works better

Store the ensemble outputs to basePath + "\Infiltration_ensembles\Dataset.ft.csv"
'''
Datasets = ["B","A","C"] #
#Dataset ="B" #for testing purposes
for Dataset in Datasets:
    if (eval("len(" + Dataset + "_FT_Accuracies) != len(" + Dataset + "_FTs)")):
        print Dataset + "_FT mismatches the accuracies list";
    actuals = "";
    results = "";
    FTs = eval(Dataset + "_FTs");
    FT_Accuracies = eval(Dataset + "_FT_Accuracies")
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
    for ft in FTs:
        #ft="datasetB_trained_clfs/Robust-LOOCV-RandomForest-MRMR-250.joblib.pkl" #for debugging only
        #load FT member name
        parts = ft.split("/")[1].split(".")[0].split("-")

        #extract clf specific indices file name
        datasetPickle = 'dataset' + Dataset + '_train' + parts[4] + '-' + parts[3] +'.joblib'

        #load clf specific indices
        clf_specific_indicies = joblib.load(basePath + 'selected_features/dataset' + Dataset + "/" + datasetPickle + ".pkl");


        #choose the subset
        #X_train = X_train_full[:, clf_specific_indicies]
        X_validate = X_validate_full[:, clf_specific_indicies]
        X_test = X_validate

        #load the classifier
        clf = joblib.load(basePath + ft);

        #retrieve accuracy from the existing records instead of trying to recalculate it.

        #predict classes
        pred = clf.predict(X_validate)
        #load CV accuracy of the clf on training datasets
        if parts[1]=="LOOCV":
            cv_size = len(pred)/2;
        else:
            cv_size = 10;
        predictions = ','.join([str(elem) for elem in pred])
        predictions = str(FT_Accuracies[FTs.index(ft)]) + "," + predictions

        #Append it to an overall results string

        results = results + predictions + "\n"
        #cross_val_score(clf, X_train, y_train, cv=cv_size, n_jobs=-1) #?? do we need this?
        #accuracy = accuracy_score(y_test, pred)#?? do we need this?
        #mcc = matthews_corrcoef(y_test, pred) #?? do we need this?
        #TODO: END OF FT LOOP

    #TODO: keep this outside the fts loop but inside the Databases loop
    #Open the file, Dump the results in the file and close it
    f=open(basePath + 'Infiltration_ensembles/ft'+Dataset+'-Ensemble.txt','a');
    f.write(Dataset + "," + actuals + "\n") #str(FTs[FTs.index(ft)]) -> Clf name
    f.write(results)
    f.close()
    # TODO: END OF Datasets LOOP
end_time = time.time();
timeTaken = end_time - start_time
#roundedResults = eval("[[0] * len(FTs)] * " + str(len(pred) + padding))
#import numpy as np

#roundedResults = np.zeros((len(pred)+2,len(FTs)))

#store the classes in the matrix to dump into the file
#roundedResults [padding:len(pred), FTs.index(ft)] = pred
# place the predictor values in the matrix roundedResults[:][0] = pred
# extract roundedResults[22][0] 22th element of 0th classifier