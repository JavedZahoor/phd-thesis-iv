import numpy
import math
import time
import json
import traceback
from MachineSpecificSettings import Settings

class DatasetLoader(object):
    s = Settings();
    __fileBasePath = s.getBasePath();
    def LoadDataSet(self, dataSetType):
        if dataSetType == "A":
            mat = scipy.io.loadmat(__fileBasePath + s.getInterimPath() + s.getDatasetAFileName());
            return mat['G0'];
        else:
            return [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]];