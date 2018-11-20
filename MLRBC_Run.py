import os
import os.path
import random

from joblib import Parallel, delayed
import pathlib
import pandas as pd
import numpy as np

from DataManagement import DataManage
import MLRBC
from config import *

class parallelRun():

    def __init__(self, numberOfExp):
        self.numberOfExperiments = numberOfExp
        self.dataFolder = DATA_FOLDER

    def doParallel(self):
        arg_instances = []
        if NUMBER_OF_FOLDS > 1:
            for it in range(NUMBER_OF_FOLDS):
                argument = []
                argument.append(it + 1)
                trainFileName = TRAIN_DATA_HEADER + "-" + str(it + 1) + ".txt"
                completeTrainFileName = os.path.join(self.dataFolder, DATA_HEADER, trainFileName)
                validFileName = VALID_DATA_HEADER + "-" + str(it + 1) + ".txt"
                completeValidFileName = os.path.join(self.dataFolder, DATA_HEADER, validFileName)
                dataManage = DataManage(completeTrainFileName, completeValidFileName)
                argument.append(dataManage)
                arg_instances.append(argument)
            Parallel(n_jobs = NO_PARALLEL_JOBS, verbose=1, backend="threading")(map(delayed(MLRBC.MLRBC), arg_instances))
        else:
            if (NO_EXPERIMENTS_AVERAGING > 1):
                trainFileName = TRAIN_DATA_HEADER + ".txt"
                completeTrainFileName = os.path.join(self.dataFolder, trainFileName)
                validFileName = VALID_DATA_HEADER + ".txt"
                completeValidFileName = os.path.join(self.dataFolder, validFileName)
                dataManage = DataManage(completeTrainFileName, completeValidFileName)
                for it in range(NO_EXPERIMENTS_AVERAGING):
                    argument = []
                    argument.append(it + 1)
                    argument.append(dataManage)
                    arg_instances.append(argument)
                Parallel(n_jobs = NO_PARALLEL_JOBS, verbose = 1, backend = "threading")(map(delayed(UCS.UCS_model), arg_instances))
            else:
                trainFileName = TRAIN_DATA_HEADER + ".txt"
                completeTrainFileName = os.path.join(self.dataFolder, DATA_HEADER, trainFileName)
                if os.path.isfile(completeTrainFileName):
                    pass
                else:
                    trainDataCSV = os.path.join(self.dataFolder, DATA_HEADER, TRAIN_DATA_HEADER + "-csv.csv")
                    convertCSV(trainDataCSV, completeTrainFileName)
                validFileName = VALID_DATA_HEADER + ".txt"
                completeValidFileName = os.path.join(self.dataFolder, DATA_HEADER, validFileName)
                if os.path.isfile(completeValidFileName):
                    pass
                else:
                    validDataCSV = os.path.join(self.dataFolder, DATA_HEADER, VALID_DATA_HEADER + "-csv.csv")
                    convertCSV(validDataCSV, completeValidFileName)
                dataManage = DataManage(completeTrainFileName, completeValidFileName)
                MLRBC.MLRBC([1, dataManage])

def convertCSV(infilename, outfilename):
    """
    :param infileName: input .csv file name
    :param outfilename: output .txt file name
    """

    try:
        df = pd.read_csv(infilename)
        Class = []
        for idx, row in df.iterrows():
            label = row[NO_ATTRIBUTES:-1]
            newlabel = "".join(map(str, label))
            Class.append(newlabel)

        labelHeader = list(df.columns)
        dfCopy = df.copy()
        classHeader = labelHeader[NO_ATTRIBUTES:]
        dfCopy.drop(classHeader, axis=1, inplace = True)
        dfCopy["Class"] = Class

        data = dfCopy.values
        print(len(data))
        headerList = list(dfCopy.columns.values)
        Header = ''
        for it in range(len(headerList)-1):
            Header = Header + headerList[it] + '\t'
        Header = Header + headerList[-1]
        np.savetxt(outfilename, data, fmt = '%s', header = Header, delimiter = '\t', newline = '\n', comments='')
    except:
        pass

if __name__== "__main__":
    random.seed(SEED_NUMBER)

    if NUMBER_OF_FOLDS < 2:
        numberOfExperiments = NO_EXPERIMENTS_AVERAGING
    else:
        numberOfExperiments = NUMBER_OF_FOLDS

    pathlib.Path(os.path.join(RUN_RESULT_PATH)).mkdir(parents=True, exist_ok=True)

    parallel = parallelRun(numberOfExperiments)
    parallel.doParallel()

    if (DO_AVERAGING == True):
        averageTrack(numberOfExperiments)