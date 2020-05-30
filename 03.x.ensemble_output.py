import numpy
import time
from time import sleep
import scipy.io

from MachineSpecificSettings import Settings

from DataSetLoaderLib import DataSetLoader
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
#from sklearn import cross_validation #it is deprecated  - use this instead from sklearn.model_selection import cross_validation
from sklearn import metrics


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import matthews_corrcoef,accuracy_score

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import LeaveOneOut


from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import StratifiedKFold


'''
- Load fitted/trained classifiers
      delme - from  this path joblib.dump(classifier,'dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-"+classifierName+'-'+method+'-'+size+'.joblib.pkl')
      classifier= joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-"+classifierName+'-'+method+'-'+size+'.joblib.pkl')
- Load test datasets
      X_test = d.LoadDataSet(dataset+"_test");
      y_test = d.LoadDataSetClasses(dataset+"_test");	
- Build ensemble [simple voting, weighted voting i.e. soft ensemble]
- For each of the row in a dataset: predict its class using the ensembles 
- calculate MCC, Time and Accuracy values for this ensemble over the whole validation dataset
- Save the results
				
'''
numpy.seterr(over='raise')
#Different feature selection methods
datasets=['B']#,'A'
methods=['MRMR'] #,'JMI','JMIM'
sizes=['10'] #,'50','100','150','200','250'
validationTechniques = ["10FoldCV"] #"LOOCV",
preps=["Standard","Robust","Quantile","Imputer"]

#Iterating over each method
print ("Dataset, prepType, validationTechnique, method, size")
for dataset in datasets:
#	f=open('mcc/mccEnsembleResults.txt','w');
#	f.write("dataset, size, method, classifier, validationTechnique, mc, timeTaken, cv.max, cv.mean, cv.min, cv.std, preprocessing");

	print "Dataset = ",dataset
	#initiating datasetloader object
	d = DataSetLoader();
	#loading relevant Data and coresponding labels of dataset A
	X_train = d.LoadDataSet(dataset+"_train");	
	y_train = d.LoadDataSetClasses(dataset+"_train");
	X_test = d.LoadDataSet(dataset+"_test");
	y_test = d.LoadDataSetClasses(dataset+"_test");		
	
	print ("Dimensions of validation data and labels:",X_test.shape,y_test.shape)
	
	#chaipee will fix it later on
	targets=list(numpy.transpose(y_train))
	y_train=[]
	for i in targets:
		y_train.append(int(i))
	

	targets=list(numpy.transpose(y_test))
	y_test=[]
	for i in targets:
		y_test.append(int(i))
	
	
	#READY with Dataset, load all the trained classifiers in a single go to make an ensemble for them
	
	for method in methods:
		#Iterating over each size
		for size in sizes:			
			for prepType in preps:
				for validationTechnique in validationTechniques:
					"""LOAD ALL TRAINED CLASSIFIERS..."""
					print (dataset + "," + prepType + "," + validationTechnique + "," +method + "," + size)
					start_time=time.time()
					mlp_pickle = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-MLP-"+method+'-'+size+'.joblib.pkl')
					svm_pickle = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-SVM-"+method+'-'+size+'.joblib.pkl')
					#adaboost = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-AdaBoost-"+method+'-'+size+'.joblib.pkl')
					#dt = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-DT-"+method+'-'+size+'.joblib.pkl')
					rf_pickle = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-RandomForest-"+method+'-'+size+'.joblib.pkl')
					#ext = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-ExtraTree-"+method+'-'+size+'.joblib.pkl')			
					print ">>>>>", mlp_pickle.best_params_
					print ">>>>>", svm_pickle.best_params_
					print ">>>>>", rf_pickle.best_params_
					
					from sklearn.ensemble import RandomForestClassifier, VotingClassifier
					"""HARD ENSEMBLE"""
					ensembleType = 'hard'
					eclf1 = VotingClassifier(estimators=[('adaboost',adaboost),('mlp', mlp),('svm', svm), ('dt',dt),('rf',rf),('ext',ext)], voting=ensembleType)
					eclf1 = eclf1.fit(X_train, y_train, 100)
					eclf1_pred = eclf1.predict(X_test)
					joblib.dump(eclf1,'dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+'-'+ensembleType+'-ensemble-'+method+'-'+size+'.joblib.pkl')
					from sklearn.metrics import *
					accuracy_score(y_test, eclf1_pred)
					matthews_corrcoef(y_test, eclf1_pred)
					"""mlp_pred = mlp.predict(X_test)
					print "mlp score after optimization...", accuracy_score(y_test, mlp_pred)
					mlp_mcc = matthews_corrcoef(y_test, mlp_pred)
					print "mlpmcc", mlp_mcc
					svm_pred = svm.predict(X_test)
					print "svm score after optimization...", accuracy_score(y_test, svm_pred)
					svm_mcc = matthews_corrcoef(y_test, svm_pred)
					print "svmmcc", svm_mcc
					"""
					mc = matthews_corrcoef(y_test,y_pred);
					end_time=time.time()-start_time		
					#now dump it in the file as CSV; dataset, size, method, classifier, validationTechnique, mc, timeTaken
					#f.write("\n "+dataset+", "+size+", "+method+", "+classifierName+", "+validationTechnique+", "+str(mc)+", "+str(end_time)+","+str(scores.max()) + "," + str(scores.mean()) + ", " + str(scores.min()) + ", " + str(scores.std() * 2)+", "+prepType);					
					print("matthew:",matthews_corrcoef(y_test, y_pred))
					print "Writing file: ", 'dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-"+classifierName+'-'+method+'-'+size+'.joblib.pkl'
					#joblib.dump(classifier,'dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-"+classifierName+'-'+method+'-'+size+'.joblib.pkl')
					sleep(0.2); 
			
#f.close()