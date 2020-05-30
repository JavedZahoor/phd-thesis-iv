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
#from sklearn.model_selection import cross_validation
from sklearn import metrics


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import matthews_corrcoef,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import LeaveOneOut


from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
#import sklearn.cross_validation

'''
	done - For each dataset, load the training and test dataset
		done - Load the FSS as per method and size
			done - for each of the classifers
				done - split training set according to validation technique i.e. n_splits=10 for 10FoldCV and n for LOOCV
				done - create a pipeline of classifier for both RAW and Normalized processing
				done - fit & calculate the training accuracy on validation part of the dataset
				done - test the 'fitted' classifier using the test dataset & record mcc value
				done - record the complete details as a row in CSV
				done - save the 'fitted' classifier for future reuse
				
'''
numpy.seterr(over='raise')
#Different feature selection methods
datasets=['C','B','A']
methods=['MRMR','JMI','JMIM']
sizes=['10','50','100','150','200','250']
classifiers = ["RandomForest","AdaBoost","DT","ExtraTree", "MLP","SVM"] 
validationTechniques = ["10FoldCV"] #"LOOCV",
preps=["Standard","Robust","Quantile","Imputer"]

basePath='' #needed when we want to run it locally
#Iterating over each method
for dataset in datasets:
	f=open('mcc/mccResults'+dataset+'.txt','a');
	f.write('\n{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ))
	#f.write("dataset, size, method, classifier, validationTechnique, mc, timeTaken, cv.max, cv.mean, cv.min, cv.std, preprocessing");
	print "Dataset = ",dataset
	#initiating datasetloader object
	d = DataSetLoader();
	#loading relevant Data and coresponding labels of dataset A
	X_train_full = d.LoadDataSet(dataset+"_train");	
	y_train = d.LoadDataSetClasses(dataset+"_train");
	X_validate_full = d.LoadDataSet(dataset+"_test");
	y_validate = d.LoadDataSetClasses(dataset+"_test");		
	
	print ("Dimensions of training data and labels:",X_train_full.shape,y_train.shape)
	print ("Dimensions of validation data and labels:",X_validate_full.shape,y_validate.shape)	
	
	
	#READY with Dataset, going to perform the main loop now
	
	for method in methods:
		#Iterating over each size
		for size in sizes:
			print ("Size and method:",size,method)
			#first run indices
			indices= joblib.load('datasetC_pickles/datasetC_train'+size+'-'+method+'.joblib.pkl')
			X_train=X_train_full[:,indices]
			X_validate=X_validate_full[:,indices]

			for prepType in preps:
				if prepType=="Standard":
					preprocess = preprocessing.StandardScaler()
				if prepType=="Robust":					
					preprocess = preprocessing.RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
				if prepType=="Quantile":
					preprocess = preprocessing.QuantileTransformer(output_distribution='normal',n_quantiles=10, random_state=0)
				if prepType=="Imputer":
					preprocess = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
				for validation in validationTechniques:
					if(validation=="LOOCV"):
						validate = X_train_full.shape[0]-15
					else:
						validate = 10
					for classifierName in classifiers:
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
							   'mlpclassifier__alpha': [1, 0.1, 0.001, 0.0001],
							   'mlpclassifier__activation': ["logistic", "relu", "tanh"],
							   'mlpclassifier__hidden_layer_sizes': [(108,1), (108,2)],
                                'mlpclassifier__max_iter': [100, 500]
							   }
							# Train the model
							grid_clf = GridSearchCV(model,param_grid,cv=validate,iid=False)
							classifier = grid_clf
						if(classifierName == "SVM"): #kernel='linear'
							model = make_pipeline(preprocess,svm.SVC( probability=True))
							Cs = [0.001, 0.01, 0.1, 1, 10]
							gammas = [0.001, 0.01, 0.1, 1]
							kernels = ['rbf','linear']
							param_grid = {'svc__C': Cs, 'svc__gamma' : gammas, 'svc__kernel':kernels}
							grid_clf = GridSearchCV(model,param_grid,cv=validate,iid=False)
							classifier = grid_clf
						if(classifierName =="AdaBoost"):
							classifier = make_pipeline(preprocess,AdaBoostClassifier())
						if(classifierName =="DT"):
							classifier = make_pipeline(preprocess,tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0))
						if(classifierName =="RandomForest"):
							model = make_pipeline(preprocess,RandomForestClassifier())					
							param_grid = {  
								'randomforestclassifier__n_estimators': [100, 300, 500],
								'randomforestclassifier__criterion': ['gini', 'entropy'],
								'randomforestclassifier__bootstrap': [True, False]
								}
							grid_clf = GridSearchCV(model,param_grid,cv=validate,iid=False)
							classifier = grid_clf					
						if(classifierName =="ExtraTree"):
							classifier = make_pipeline(preprocess,ExtraTreesClassifier(n_estimators=10, max_depth=10000,min_samples_split=2, random_state=0))
						
						classifier.fit(X_train,y_train)
						extra_info = ""
						for validationTechnique in validationTechniques:
							if validationTechnique=="LOOCV":						
								#https://www.programcreek.com/python/example/91872/sklearn.cross_validation.LeaveOneOut
								folds = cross_validation.LeaveOneOut(X_train.shape[0])
								scores = cross_val_score(classifier, X_train, y_train, cv=folds, n_jobs=-1)
							else:
								folds = 10
								scores = cross_val_score(classifier, X_train, y_train, cv=folds)
							print scores.max(), scores.mean(), scores.min(), scores.std() * 2
							if(classifierName == "MLP" or classifierName == "SVM" or classifierName=="RandomForest"):
								y_pred = classifier.best_estimator_.predict(X_validate)
								extra_info = classifier.best_params_
							else:
								y_pred = classifier.predict(X_validate);
							print "------Validation Accuracy-------"
							print y_pred.shape					
							print y_pred
							
							print numpy.array(y_validate).shape
							print y_validate
							#transform data to be ready for mcc
							y_pred=numpy.array(y_pred)
							y_pred[y_pred == 0] = -1
							
							y_validate=numpy.array(y_validate)					
							y_validate[y_validate == 0] = -1
						
							mc = matthews_corrcoef(y_validate,y_pred);
							cm = confusion_matrix (y_validate,y_pred);
							"""
							Add CM to the list of outputs
							"""
							end_time=time.time()-start_time		
							#now dump it in the file as CSV; dataset, size, method, classifier, validationTechnique, mc, timeTaken
							f.write("\n "+dataset+", "+size+", "+method+", "+classifierName+", "+validationTechnique+", "+str(mc)+", "+str(end_time)+","+str(scores.max()) + "," + str(scores.mean()) + ", " + str(scores.min()) + ", " + str(scores.std() * 2)+", "+prepType);					
							print("matthew:",matthews_corrcoef(y_validate, y_pred))
							print "Writing file: ", 'dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-"+classifierName+'-'+method+'-'+size+'.joblib.pkl'
							joblib.dump(classifier,'dataset'+dataset+'_trained_clfs/'+prepType+"-"+validationTechnique+"-"+classifierName+'-'+method+'-'+size+'.joblib.pkl')
							sleep(0.2); 
			
f.close()