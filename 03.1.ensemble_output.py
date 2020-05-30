import numpy
import time
from time import sleep
import scipy.io

from MachineSpecificSettings import Settings
from sklearn.metrics import *
from DataSetLoaderLib import DataSetLoader
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

'''
    done - For each dataset, load the training and test dataset
    done - load the trained classifiers
    done - setup the pipeline
    done - use best_params from the optimized classifiers
    - run all 3 ensembles
        - use only the reliable classifiers in the ensemble for hard/soft ensembles
        - for weighted ensemble use mcc * score > 0 as a measure to include the classifier
    - log the findings in a file
                
'''
numpy.seterr(over='raise')
#Different feature selection methods
datasets=['C','B','A']
methods=['MRMR','JMI','JMIM']
sizes=['10','50','100','150','200','250']
classifiers = ["MLP","SVM","RandomForest","AdaBoost","DT","ExtraTree"]
validationTechniques = ["10FoldCV"] #,"LOOCV"
preps=["Standard","Robust","Quantile","Imputer"]
ensemble_types=['hard','soft']#,'weighted'
n_iter_search = 20

#Iterating over each method
for dataset in datasets:
    f=open('mcc/mccResults'+dataset+'-Ensemble.txt','a');
    f.write("dataset, size, method, classifier, validationTechnique, mc, timeTaken, score, preprocessing, params\n");
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

    #chaipee will fix it later on

    targets=list(numpy.transpose(y_train))
    y_train=[]
    for i in targets:
        y_train.append(int(i))

    targets=list(numpy.transpose(y_validate))
    y_validate=[]
    for i in targets:
        y_validate.append(int(i))
    
    
    #READY with Dataset, going to perform the main loop now
    
    for method in methods:
        #Iterating over each size
        for size in sizes:
            print ("Size and method:",size,method)
            #first run indices
            indices= joblib.load('selected_features/dataset'+str(dataset)+'/dataset'+str(dataset)+'_train'+str(size)+'-'+str(method)+'.joblib.pkl')
            X_train=X_train_full[:,indices]
            X_validate=X_validate_full[:,indices]
            X_test = X_validate
            y_test = y_validate
            preprocess = None
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
                        validate = LeaveOneOut()
                        folds = LeaveOneOut(X_train.shape[0])
                    else:
                        validate = 10
                        folds = 10

                    start_time=time.time();
                    validation = "10FoldCV"                    
                    mlp_model = make_pipeline(preprocess,MLPClassifier(solver='adam',alpha=0.0001,activation='relu',batch_size=150,hidden_layer_sizes=(200, 100),random_state=1))
                    mlp_pickle = joblib.load('dataset'+str(dataset)+'_trained_clfs/'+str(prepType)+'-'+str(validation)+'-MLP-'+str(method)+'-'+str(size)+'.joblib.pkl')
                    #param_grid=mlp_pickle.best_params_
                    # Train the model
                    mlp = mlp_pickle.best_estimator_ #RandomizedSearchCV(mlp_model,param_grid,cv=validate,iid=False,n_iter=n_iter_search, n_jobs=-1)
                    mlp_scores = cross_val_score(mlp, X_train, y_train, cv=folds, n_jobs=-1)
                    mlp_pred = mlp.predict(X_validate)
                    mlp_accuracy = accuracy_score(y_test, mlp_pred)
                    mlp_mcc = matthews_corrcoef(y_test, mlp_pred)
                    end_time = time.time();
                    timeTaken = end_time - start_time
                    extra_info = str(mlp_pickle.best_params_)
                    f.write(
                        dataset + "," + str(size)+ "," +method+ ",MLP," +"10FoldCV"+ "," + str(mlp_mcc)+ "," +str(timeTaken)+ "," +str(mlp_accuracy)+ "," +prepType + "," + extra_info + '\n');
                    start_time = time.time();

                    svm_model = make_pipeline(preprocess)
                    svm_pickle = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validation+"-SVM-"+method+'-'+size+'.joblib.pkl')
                    #param_grid = svm_pickle.best_params_
                    svm = svm_pickle.best_estimator_ #RandomizedSearchCV(svm_model,param_grid,cv=validate,iid=False,n_iter=n_iter_search, n_jobs=-1)
                    svm_scores = cross_val_score(svm, X_train, y_train, cv=folds, n_jobs=-1)
                    svm_pred = svm.predict(X_validate)
                    svm_accuracy = accuracy_score(y_test, svm_pred)
                    svm_mcc = matthews_corrcoef(y_test, svm_pred)
                    end_time = time.time();
                    timeTaken = end_time - start_time
                    f.write(
                        dataset + "," + str(size) + "," + method + ",SVM," + validation + "," + str(
                            svm_mcc) + "," + str(timeTaken) + "," + str(svm_accuracy) + "," + prepType + "," + str(svm_pickle.best_params_)+ '\n');
                    start_time = time.time();

                    ab = make_pipeline(preprocess,AdaBoostClassifier())
                    ab.fit(X_train, y_train)
                    ab_scores = cross_val_score(ab, X_train, y_train, cv=folds, n_jobs=-1)
                    ab_pred = ab.predict(X_validate)
                    ab_accuracy = accuracy_score(y_test, ab_pred)
                    ab_mcc = matthews_corrcoef(y_test, ab_pred)
                    end_time = time.time();
                    timeTaken = end_time - start_time
                    f.write(
                        dataset + "," + str(size) + "," + method + ",AdaBoost," + validation + "," + str(
                            mlp_mcc) + "," + str(timeTaken) + "," + str(mlp_accuracy) + "," + prepType + ",-"+ '\n');
                    start_time = time.time();

                    dt = make_pipeline(preprocess,tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0))
                    dt.fit(X_train, y_train)
                    dt_scores = cross_val_score(dt, X_train, y_train, cv=folds, n_jobs=-1)
                    dt_pred = dt.predict(X_validate)
                    dt_accuracy = accuracy_score(y_test, dt_pred)
                    dt_mcc = matthews_corrcoef(y_test, dt_pred)
                    end_time = time.time();
                    timeTaken = end_time - start_time
                    f.write(
                        dataset + "," + str(size) + "," + method + ",DecisionTrees," + validation + "," + str(
                            dt_mcc) + "," + str(timeTaken) + "," + str(dt_accuracy) + "," + prepType + ",-"+ '\n');
                    start_time = time.time();

                    rf_model = make_pipeline(preprocess,RandomForestClassifier())
                    rf_pickle = joblib.load('dataset'+dataset+'_trained_clfs/'+prepType+"-"+validation+"-RandomForest-"+method+'-'+size+'.joblib.pkl')
                    #param_grid = rf_pickle.best_params_
                    rf = rf_pickle.best_estimator_ #RandomizedSearchCV(rf_model,param_grid,cv=validate,iid=False,n_iter=n_iter_search, n_jobs=-1)
                    rf.fit(X_train, y_train)
                    rf_scores = cross_val_score(rf, X_train, y_train, cv=folds, n_jobs=-1)
                    rf_pred = rf.predict(X_validate)
                    rf_accuracy = accuracy_score(y_test, rf_pred)
                    rf_mcc = matthews_corrcoef(y_test, rf_pred)
                    end_time = time.time();
                    timeTaken = end_time - start_time
                    f.write(
                        dataset + "," + str(size) + "," + method + ",RandomForest," + validation + "," + str(
                            rf_mcc) + "," + str(timeTaken) + "," + str(rf_accuracy) + "," + prepType+"," + str(rf_pickle.best_params_)+ '\n');
                    start_time = time.time();

                    et = make_pipeline(preprocess,ExtraTreesClassifier(n_estimators=10, max_depth=10000,min_samples_split=2, random_state=0))
                    et.fit(X_train,y_train)
                    et_scores = cross_val_score(et, X_train, y_train, cv=folds, n_jobs=-1)
                    et_pred = et.predict(X_validate)
                    et_accuracy = accuracy_score(y_test, et_pred)
                    et_mcc = matthews_corrcoef(y_test, et_pred)
                    end_time = time.time();
                    timeTaken = end_time - start_time
                    f.write(
                        dataset + "," + str(size) + "," + method + ",ExtraTrees," + validation + "," + str(
                            et_mcc) + "," + str(timeTaken) + "," + str(et_accuracy) + "," + prepType + ",-"+ '\n');
                    start_time = time.time();

                    extra_info = ""
                    #print scores.max(), scores.mean(), scores.min(), scores.std() * 2
                    M = max([mlp_mcc,svm_mcc, et_mcc, rf_mcc, dt_mcc, ab_mcc])
                    ests = [] #estimators that we are going to use in the ensemble
                    if M > 0: # we do have some good classifiers so we will NOT include the ones with mcc <=0
                        if mlp_mcc > 0:
                            ests.append(('mlp', mlp))
                        if svm_mcc >0:
                            ests.append(('svm', svm))
                        if rf_mcc>0:
                            ests.append(('rf',rf))
                        if et_mcc>0:
                            ests.append(('et',et))
                        if dt_mcc>0:
                            ests.append(('dt',dt))
                        if ab_mcc>0:
                            ests.append(('ab',ab))
                    else: #use all of them, all of them ar equally bad
                        ests.append(('mlp', mlp))
                        ests.append(('svm', svm))
                        ests.append(('rf',rf))
                        ests.append(('et',et))
                        ests.append(('dt',dt))
                        ests.append(('ab',ab))

                    if (len(ests) > 0): #lets run the ensemble and record its efficiency now...
                        for ens_type in ensemble_types:
                            eclf1 = VotingClassifier(estimators=ests, voting=ens_type)
                            eclf1 = eclf1.fit(X_train, y_train)
                            eclf1_pred = eclf1.predict(X_test)
                            ensemble_accuracy = accuracy_score(y_test, eclf1_pred)
                            ensemble_mcc = matthews_corrcoef(y_test, eclf1_pred)
                            end_time = time.time();
                            timeTaken = end_time - start_time
                            f.write(
                                dataset + "," + str(size) + "," + method + ",Ensemble-"+ens_type+"," + validation + "," + str(
                                    ensemble_mcc) + "," + str(timeTaken) + "," + str(ensemble_accuracy) + "," + prepType+",-"+ '\n');

                            print "ENSEMBLE score after...", ensemble_accuracy, " mcc", ensemble_mcc
                            joblib.dump(eclf1, 'dataset' + dataset + '_trained_clfs/' + prepType + "-" + str(validate) + "-" + 'Ensemble-'+ ens_type +  '-' + method + '-' + str(size) + '.joblib.pkl')


                    """
                    Add CM to the list of outputs
                    """

                    #sleep(0.2);
            
f.close()