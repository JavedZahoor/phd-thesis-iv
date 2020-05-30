import sys
sys.path.insert(0, '/home/javedzahoor/research/mattia/code/jz')
import threading
import time
import ref_4_mifs as mifs
from sklearn.externals import joblib
def SelectSubSetmRMR(vectors, classes, useMethod='MRMR'):#, numOfFeatures=500):
	X = vectors
	y = classes
	# define MI_FS feature selection method
	feat_selector = mifs.MutualInformationFeatureSelector(method=useMethod) #n_features=numOfFeatures)

	# find all relevant features
	feat_selector.fit(X, y)

	# check selected features
	feat_selector.support_

	# check ranking of features
	return feat_selector.ranking_

"""
TEST PROGRAM
"""
from GlobalUtils import *
import scipy
from MachineSpecificSettings import Settings
import scipy.io
import numpy
from DataSetLoaderLib import DataSetLoader
import csv



values=[]
class myThread (threading.Thread):
    def __init__(self, threadID, name,add_by, variables,targets):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.variables=variables
        self.targets=targets
        self.add_by=add_by
    def run(self):
		threadLock.acquire()
		try:
			methods=['MRMR','JMI','JMIM']
			for useMethod in methods:
				#useMethod = 'MRMR'
				print "Starting " + self.name
				print self.add_by
				x=SelectSubSetmRMR(self.variables, self.targets, useMethod)
	        		for i in x:
					if i+self.add_by>=545089:
						i+=1
					values.append(i+self.add_by)
				joblib.dump(values,'selected_features/datasetC/selected_indices'+'_'+useMethod+'.joblib.pkl', compress=9) 
		except:
			type, value, traceback = sys.exc_info()
			print('Error Occured %s: %s: %s' % (type, value, traceback))
		threadLock.release()
		print len(values)
        	print "Exiting " + self.name
		return
		

threads=[]

d = DataSetLoader();
G = d.LoadDataSet("C_train");
targets = d.LoadDataSetClasses("C_train");
#chaipee will fix it later on
#y_train=numpy.transpose(targets)
y_train=numpy.asarray(targets)
print y_train.shape
targets=list(y_train)
y_train=[]
for i in targets:
	print i
	y_train.append(int(i))
targets = y_train
#targets =numpy.asarray(targets )
#targets = column_or_1d(targets, warn=True)

print "Dataset loaded"

G=numpy.asarray(G)
threadLock = threading.Lock()
print G.shape
vals=649
original=649
for i in range(0,695556):
    print "vals= "+str(vals)+"\n"
    # Create new threads
    thread = myThread(i, "Thread-"+str(i),vals-original,G[:,vals-original:vals],targets)
    
    # Add threads to thread list
    threads.append(thread)
    vals+=original
# Wait for all threads to complete
for t in threads:
    # Start new Threads
    t.start()
    t.join()
#Saving the selected_indices



print "Exiting Main Thread"
