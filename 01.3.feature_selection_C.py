from GlobalUtils import *
import ref_4_mifs as mifs
import ref_4_mi as mi
import sendemail as EMAIL
from sklearn.externals import joblib
import sklearn
import numpy
from MachineSpecificSettings import Settings
from DataSetLoaderLib import DataSetLoader
import time

@timing
def SelectSubSetmRMR(vectors, classes,useMethod,features):
	X = vectors
	y = classes
	# define MI_FS feature selection method
	feat_selector = mifs.MutualInformationFeatureSelector(method=useMethod,n_features=features)

	# find all relevant features
	feat_selector.fit(X, y)

	# check selected features
	print (feat_selector.support_)

	
	# check ranking of features
	print (feat_selector.ranking_)
	print (len(feat_selector.ranking_))
	selected_indices=feat_selector.ranking_

	# call transform() on X to filter it down to selected features
	X_filtered = feat_selector.transform(X)
	return [X_filtered,selected_indices]

@timing
def loadDataset(identifier):
    d = DataSetLoader()
    x = d.LoadDataSet(identifier)
    print 'X', x.shape
    y= d.LoadDataSetClasses(identifier)
    print 'Y', y.shape
    #y=numpy.transpose(y.astype(numpy.int64))
    y = sklearn.utils.validation.column_or_1d(y, warn=True)
    print 'Y', y.shape
    target=[]
    y=list(y)
    print "y before manual transform =" , y
    for i in y:
        target.append(int(i))
    print len(y)
    print y
    return x, y

@timing
def mainloop():
    datasets=['C_train']
    sizes=['10','50','100','150','200','250']
    methods=['MRMR','JMI','JMIM']
    for dataset in datasets:
        x, y = loadDataset(dataset)
        for method in methods:
            for size in sizes:
	        print size
                print method
                selected_indices=[]
                #return
                [subset,selected_indices] = SelectSubSetmRMR(x,y,method,int(size))
                joblib.dump(selected_indices,'selected_features/dataset' + str(dataset) + str(size) + '-' + method + '.joblib.pkl', compress=9) 
                print "Saved new selected indices"
                EMAIL.SendEmail(' DONE',str(dataset) + str(size) + '-' + method + '.joblib.pkl' )

mainloop()