import os
import os.path
import random

from joblib import Parallel, delayed
import pathlib
import pandas as pd
import numpy as np
from math import sqrt

from DataManagement import DataManage
import MLRBC
from config import *

class parallelRun():

    def __init__(self, numberOfExp):
        self.numberOfExperiments = numberOfExp

    def doParallel(self):
        arg_instances = []
        if NUMBER_OF_FOLDS > 1:
            for it in range(NUMBER_OF_FOLDS):
                argument = []
                argument.append(it + 1)
                trainFileName = TRAIN_DATA_HEADER + "-" + str(it + 1) + ".txt"
                completeTrainFileName = os.path.join(DATA_FOLDER, DATA_HEADER, trainFileName)
                validFileName = VALID_DATA_HEADER + "-" + str(it + 1) + ".txt"
                completeValidFileName = os.path.join(DATA_FOLDER, DATA_HEADER, validFileName)
                dataManage = DataManage(completeTrainFileName, completeValidFileName)
                argument.append(dataManage)
                arg_instances.append(argument)
            Parallel(n_jobs = NO_PARALLEL_JOBS, verbose=1, backend="threading")(map(delayed(MLRBC.MLRBC), arg_instances))
        else:
            if (NO_EXPERIMENTS_AVERAGING > 1):
                trainFileName = TRAIN_DATA_HEADER + ".txt"
                completeTrainFileName = os.path.join(DATA_FOLDER, trainFileName)
                validFileName = VALID_DATA_HEADER + ".txt"
                completeValidFileName = os.path.join(DATA_FOLDER, validFileName)
                dataManage = DataManage(completeTrainFileName, completeValidFileName)
                for it in range(NO_EXPERIMENTS_AVERAGING):
                    argument = []
                    argument.append(it + 1)
                    argument.append(dataManage)
                    arg_instances.append(argument)
                Parallel(n_jobs = NO_PARALLEL_JOBS, verbose = 1, backend = "threading")(map(delayed(UCS.UCS_model), arg_instances))
            else:
                trainFileName = TRAIN_DATA_HEADER + ".txt"
                completeTrainFileName = os.path.join(DATA_FOLDER, DATA_HEADER, trainFileName)
                if os.path.isfile(completeTrainFileName):
                    pass
                else:
                    trainDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-csv.csv")
                    convertCSV(trainDataCSV, completeTrainFileName)
                validFileName = VALID_DATA_HEADER + ".txt"
                completeValidFileName = os.path.join(DATA_FOLDER, DATA_HEADER, validFileName)
                if os.path.isfile(completeValidFileName):
                    pass
                else:
                    validDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-csv.csv")
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
            label = [int(l) for l in row[NO_ATTRIBUTES:-1]]
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

def dataProp(infilename):
    """
    :param infilename:
    :return:
    """
    df = pd.read_csv(infilename)
    Class = []
    for idx, row in df.iterrows():
        label = [int(l) for l in row[NO_ATTRIBUTES:-1]]
        newlabel = "".join(map(str, label))
        Class.append(newlabel)

    classCount = len(Class[0])
    dataCount = len(Class)
    labelHeader = list(df.columns)
    dfCopy = df.copy()
    classHeader = labelHeader[NO_ATTRIBUTES:]
    dfCopy.drop(classHeader, axis=1, inplace = True)
    dfCopy["Class"] = Class
    data = dfCopy

    count = 0.0
    for rowIdx, row in data.iterrows():
        label = row["Class"]
        count += countLabel(label)
    card = count / dataCount
    dens = card / classCount

    Y = np.empty([classCount])
    for y in range(classCount):
        sampleCount = 0
        for rowIdx, row in data.iterrows():
            label = row["Class"]
            if label[y] == '1':
                sampleCount += 1
        Y[y] = sampleCount

    IRLbl = np.empty([classCount])
    maxIR = Y.max()
    for it in range(classCount):
        IRLbl[it] = (maxIR/Y[it])
    meanIR = IRLbl.sum() / classCount

    temp = (IRLbl - meanIR)**2
    IRLbls = sqrt(temp.sum() / (classCount - 1))
    CVIR = IRLbls / meanIR

    dataInfo = dict(zip(["card", "dens", "MaxIR", "MeanIR", "IRLbls", "CVIR"], [card, dens, maxIR, meanIR, IRLbls, CVIR]))
    print(dataInfo)

def countLabel(label):
    count = 0
    for L in label:
        if float(L) != 0:
            count += 1
    return count

if __name__== "__main__":
    random.seed(SEED_NUMBER)
    DataFileName = DATA_HEADER + "-csv.csv"
    completeDataFileName = os.path.join(DATA_FOLDER, DATA_HEADER, DataFileName)
    # dataProp(completeDataFileName)

    if NUMBER_OF_FOLDS < 2:
        numberOfExperiments = NO_EXPERIMENTS_AVERAGING
    else:
        numberOfExperiments = NUMBER_OF_FOLDS

    pathlib.Path(os.path.join(RUN_RESULT_PATH)).mkdir(parents=True, exist_ok=True)

    parallel = parallelRun(numberOfExperiments)
    parallel.doParallel()

    if (DO_AVERAGING == True):
        averageTrack(numberOfExperiments)