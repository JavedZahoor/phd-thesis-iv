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
validationTechnique = ['LOOCV',"10FoldCV"]
preprocessing = ['Standard','Imputer','Robust','Quantile']
#datasets = ["A","B"]
classifiers = ["MLP","SVM","AdaBoost","DT","RandomForest","ExtraTree"]
dataset = "A"
f=open('mcc/mccResults'+dataset+'.txt','w');
f.write("dataset, size, method, classifier, validationTechnique, mc, timeTaken, extra info");

for classifierName in classifiers:
    for method in methods:
        for size in sizes:
			for preproc in preprocessing:
				d = DataSetLoader();
				X_train= d.LoadDataSet(dataset);
				y_train = d.LoadDataSetClasses(dataset);
				#print X_train.shape
				#print y_train.shape
				#chaipee will fix it later on
				y_train=numpy.transpose(y_train)
				#print y_train.shape
				targets=list(y_train)
				y_train=[]
				for i in targets:
					#print i
					y_train.append(int(i))
				#print len(y_train)
				
				#first run indices				
				indices= joblib.load('dataset'+dataset+'_pickles/selected_indices_'+method+'.joblib.pkl')
				X_train=X_train[:,indices]
				
				#second run indices
				indices= joblib.load('dataset'+dataset+'_pickles/'+size+'-'+method+'.joblib.pkl')
				X_train=X_train[:,indices]
				if(preproc== "Standard"):
					preprocess = preprocessing.StandardScaler()
				elif(preproc== "Imputer"):
					preprocess = preprocessing.Imputer()
				elif(preproc== "Quantile"):
					preprocess = preprocessing.QuantileTransformer(n_quantiles=10, random_state=0)
				elif(preproc== "Robust"):
					preprocess = preprocessing.RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True)
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
					classifier = make_pipeline(preprocess,AdaBoostClassifier())
				if(classifierName =="DT"):
					classifier = make_pipeline(preprocess,tree.DecisionTreeClassifier())
				if(classifierName =="RandomForest"):
					classifier = make_pipeline(preprocess,RandomForestClassifier())
					"""
						grid_param = {  
						'n_estimators': [100, 300, 500, 800, 1000],
						'criterion': ['gini', 'entropy'],
						'bootstrap': [True, False]
						}
					"""
				if(classifierName =="ExtraTree"):
					classifier = make_pipeline(preprocessing.StandardScaler(),ExtraTreesClassifier())            

				classifier.fit(X_train,y_train);
				extra_info = ""
				if(classifierName == "MLP" or classifierName == "SVM"):
					p = classifier.best_estimator_.predict(X_train)
					extra_info = classifier.best_params_
				else:
					p=classifier.predict(X_train);
				mc = matthews_corrcoef(y_train,p);
				end_time=time.time()-start_time		
				#TODO: SAVE THE TRAINGED CLASSIFIER AS WELL FOR LATER USER
				#now dump it in the file as CSV; dataset, size, method, classifier, validationTechnique, mc, timeTaken
				f.write("\n A, "+size+", "+method+", "+classifierName+", LOOCV, "+str(mc)+", "+str(end_time)+", " + extra_info);
				sleep(0.2); 
f.close()