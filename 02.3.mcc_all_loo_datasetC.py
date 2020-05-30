from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import matthews_corrcoef
from sklearn.externals import joblib
import scipy.io
from time import sleep
import numpy
import time

from sklearn import datasets
from sklearn.model_selection import cross_val_score

from DataSetLoaderLib import DataSetLoader
from MachineSpecificSettings import Settings

import numpy as np

""" imports for pipeline """
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import *
""" imports for ensemble of classifiers """
#add headers
sizes = ['10','50','100','150','200','250']
methods = ['MRMR','JMI','JMIM']
#validationTechnique = ['LOOCV',"10FoldCV"] -- NOT USED???
#preprocessing = ['','NP']
#datasets = ["A","B"]
classifiers = ["MLP","SVM","AdaBoost","DT","RandomForest","ExtraTree"]

f=open('mcc/mccResultsC.txt','w');
f.write("dataset, size, method, classifier, validationTechnique, mc, timeTaken");

for classifierName in classifiers:
    for method in methods:
        for size in sizes:            
            #print size
            #print method        
            d = DataSetLoader();
            X_train= d.LoadDataSet("C_train");
            y_train = d.LoadDataSetClasses("C_train");
            X_test= d.LoadDataSet("C_test");
            y_test = d.LoadDataSetClasses("C_test");

            #chaipee will fix it later on
            y_train=numpy.transpose(y_train)
	    print y_train.shape
	    targets=list(y_train)
	    y_train=[]
	    for i in targets:
		#print i
		y_train.append(int(i))

            y_test=numpy.transpose(y_train)
	    print y_test.shape
	    targets=list(y_train)
	    y_test=[]
	    for i in targets:
		#print i
		y_test.append(int(i))
            #first run indices
            #TODO: replace A with dataset variable, Also add NP etc. and validationTechnique
            indices= joblib.load('datasetC_pickles/datasetC'+size+'-'+method+'.joblib.pkl')
            X_train=X_train[:,indices]
            X_test=X_test[:,indices]
            print "-------------"
            print X_test.shape
            print "-------------"
	    preprocess = preprocessing.StandardScaler()
	    #TODO: Cater for other preprocessors here
            start_time=time.time()
    	    if(classifierName == "MLP"):                
				model = make_pipeline(
					preprocess,
					MLPClassifier(solver='adam',
								  alpha=0.0001,
								  activation='relu',
									batch_size=150,
									hidden_layer_sizes=(200, 100),
									random_state=1))
					classifier = make_pipeline(preprocess, MLPClassifier(activation='logistic',solver='sgd'))
				# Construct the parameter grid	
				param_grid={
				   'mlpclassifier__learning_rate': ["constant", "invscaling", "adaptive"],   
				   'mlpclassifier__alpha': [1, 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
				   'mlpclassifier__activation': ["logistic", "relu", "tanh"],
				   'mlpclassifier__hidden_layer_sizes': [(100,1), (100,2), (100,3)]   
				   }
				# Train the model
				grid_clf = GridSearchCV(model,param_grid,cv=10,iid=False)
				classifier = grid_clf
    	    if(classifierName == "SVM"):
    	        model = make_pipeline(preprocess,svm.SVC( probability=True))


				Cs = [0.001, 0.01, 0.1, 1, 10]
				gammas = [0.001, 0.01, 0.1, 1]
				kernels = ['rbf','linear']
				param_grid = {'svc__C': Cs, 'svc__gamma' : gammas, 'svc__kernel':kernels}
				grid_clf = GridSearchCV(model,param_grid,cv=10,iid=False)
				classifier = grid_clf
    	    if(classifierName =="AdaBoost"):
    	        classifier = make_pipeline(preprocessing.StandardScaler(),AdaBoostClassifier())
    	    if(classifierName =="DT"):
    	        classifier = make_pipeline(preprocessing.StandardScaler(),tree.DecisionTreeClassifier())
    	    if(classifierName =="RandomForest"):
    	        classifier = make_pipeline(preprocessing.StandardScaler(),RandomForestClassifier())
    	    if(classifierName =="ExtraTree"):
    	        classifier = make_pipeline(preprocessing.StandardScaler(),ExtraTreesClassifier())
            

            classifier.fit(X_train,y_train);
			extra_info = ""
			if(classifierName == "MLP" || classifierName == "SVM"):
				p = classifier.best_estimator_.predict(X_train)
				extra_info = classifier.best_params_
            else:
				p=classifier.predict(X_train);
            mc = matthews_corrcoef(y_train,p);
            end_time=time.time()-start_time		
			#TODO: SAVE THE TRAINGED CLASSIFIER AS WELL FOR LATER USER
            #now dump it in the file as CSV; dataset, size, method, classifier, validationTechnique, mc, timeTaken
            f.write("\n C, "+size+", "+method+", "+classifierName+", LOOCV, "+str(mc)+", "+str(end_time)+", " + extra_info);
            sleep(0.2); 
f.close()