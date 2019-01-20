import os
import os.path
import random

from joblib import Parallel, delayed
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split

from DataManagement import DataManage
import MLRBC
from config import *

class averageTrack():

    def __init__(self, NumberOfExperiment):
        self.NumberOfExperiments = NumberOfExperiment
        self.outFileHeader = 'TRMC_UCS_averageTrack'

        self.aveTrack()
        # self.saveAverage()
        # self.avePerformance()

    def aveTrack(self):
        datasetList = []  # np.array([])
        for exp in range(self.NumberOfExperiments):
            file_name = DATA_HEADER  + "_MLRBC_LearnTrack" + "_" + str(exp+1) + '.txt'
            completeName = os.path.join(RUN_RESULT_PATH, file_name)
            try:
                arraylist = np.array([])
                headerList = np.array([])
                ds = open(completeName, 'r')
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', file_name)
                raise
            else:
                headerList = ds.readline().rstrip('\n').split('\t')  # strip off first row
                for line in ds:
                    lineList = line.strip('\n').split('\t')
                    arraylist = [float(i) for i in lineList]
                    datasetList.append(arraylist)
                ds.close()

        # print(datasetList)
        a = [row[0] for row in datasetList]  # Iterations
        b = [row[1] for row in datasetList]  # macro population size
        c = [row[2] for row in datasetList]  # micro population size
        d = [row[3] for row in datasetList]  # Hamming loss
        e = [row[4] for row in datasetList]  # Accuracy
        f = [row[5] for row in datasetList]  # Generality
        g = [row[7] for row in datasetList]  # TP
        h = [row[7] for row in datasetList]  # TN

        self.IterNum = a[0:int(len(a) / self.NumberOfExperiments)]

        if PLOT_SETTING[0]:
            macroPopSize = []
            self.macroPopSize_ave = []
            for i in range(self.NumberOfExperiments):
                macroPopSize.append((b[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
            macroPopSize = np.sum(macroPopSize, axis=0)
            for x in macroPopSize:
                self.macroPopSize_ave.append(x / self.NumberOfExperiments)

            microPopSize = []
            self.microPopSize_ave = []
            for i in range(self.NumberOfExperiments):
                microPopSize.append((c[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
            microPopSize = np.sum(microPopSize, axis=0)
            for x in microPopSize:
                self.microPopSize_ave.append(x / self.NumberOfExperiments)

            plt.figure(1)
            plt.plot(self.IterNum, self.macroPopSize_ave, 'r-', label='MacroPopSize')
            plt.plot(self.IterNum, self.microPopSize_ave, 'b-', label='MicroPopSize')
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

        if PLOT_SETTING[1]:
            accuracyEstimate = []
            self.accuracyEstimate_ave = []
            for i in range(self.NumberOfExperiments):
                accuracyEstimate.append((e[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
            accuracyEstimate = np.sum(accuracyEstimate, axis=0)
            for x in accuracyEstimate:
                self.accuracyEstimate_ave.append(x / self.NumberOfExperiments)

            plt.figure(2)
            plt.plot(self.IterNum, self.accuracyEstimate_ave, 'b-', label = 'Accuracy Estimate')
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

        if PLOT_SETTING[2]:
            hlossEstimate = []
            self.hlossEstimate_ave = []
            for i in range(self.NumberOfExperiments):
                hlossEstimate.append((d[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
            hlossEstimate = np.sum(hlossEstimate, axis=0)
            for x in hlossEstimate:
                self.hlossEstimate_ave.append(x / self.NumberOfExperiments)

            plt.figure(3)
            plt.plot(self.IterNum, self.hlossEstimate_ave, '-b', label = "Hamming loss")
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

        if PLOT_SETTING[3]:
            aveGenerality = []
            self.aveGenerality_ave = []
            for i in range(self.NumberOfExperiments):
                aveGenerality.append((f[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
            aveGenerality = np.sum(aveGenerality, axis=0)
            for x in aveGenerality:
                self.aveGenerality_ave.append(x / self.NumberOfExperiments)

            plt.figure(4)
            plt.plot(self.IterNum, self.aveGenerality_ave, 'b-', label='AveGenerality')
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

        if PLOT_SETTING[4]:
            TP = []
            self.tp_ave = []
            for i in range(self.NumberOfExperiments):
                TP.append((g[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
            TP = np.sum(TP, axis=0)
            for x in TP:
                self.tp_ave.append(x / self.NumberOfExperiments)

            TN = []
            self.tn_ave = []
            for i in range(self.NumberOfExperiments):
                TN.append((h[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
            TN = np.sum(TN, axis=0)
            for x in TN:
                self.tn_ave.append(x / self.NumberOfExperiments)

            plt.figure(5)
            plt.plot(self.IterNum, self.tp_ave, 'b-', label='TP')
            plt.plot(self.IterNum, self.tn_ave, 'r-', label='TN')
            legend = plt.legend(loc='center', shadow=True, fontsize='small')
            plt.xlabel('Iteration')
            plt.ylim(bottom = 0.0)

        plt.show()

    def saveAverage(self):

        file_name = self.outFileHeader + '.txt'
        completeName = os.path.join(MAIN_RESULTS_PATH, self.aveResultPath, file_name)
        try:
            learnTrackOut = open(completeName, 'w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', file_name)
            raise
        else:
            learnTrackOut.write("Explore_Iteration\tMacroPopSize\tMicroPopSize\tAccuracy_Estimate\tAveGenerality\n")

        for i in range(len(self.IterNum)):
            trackString = str(self.IterNum[i]) + "\t" + str(self.macroPopSize_ave[i]) + "\t" + str(self.microPopSize_ave[i]) + "\t" + str(
                self.accuracyEstimate_ave[i]) + "\t" + str("%.2f" % self.aveGenerality_ave[i]) + "\n"
            learnTrackOut.write(trackString)

        learnTrackOut.close()

    def avePerformance(self):
        n = 3
        performance = {}
        performanceList = []
        for exp in range(self.NumberOfExperiments):
            file_name = POP_STAT_HEADER + "_" + str(exp + 1) + '.txt'
            completeName = os.path.join(MAIN_RESULTS_PATH, POP_STAT_PATH, file_name)
            try:
                headerList = np.array([])
                f = open(completeName, 'r')
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open ', file_name)
                raise
            else:
                title = f.readline()
                headerList = f.readline().rstrip('\n').split('\t')  # strip off first row
                for i in range(n):
                    lineList = f.readline().strip('\n').split('\t')
                    perf = []
                    for p in lineList[1:]:
                        if p != 'NA':
                            perf.append(float(p))
                        else:
                            perf.append(0.0)
                    performance[lineList[0]] = perf
                f.close()
            performanceList.append(performance)

        avePerformance = dict.fromkeys(performanceList[0].keys(), np.array([0, 0]))
        for prf in performanceList:
            for key in prf.keys():
                array = np.array(prf[key])
                avePerformance[key] = avePerformance[key] + array

        for key in avePerformance.keys():
            avePerformance[key] = avePerformance[key] / self.NumberOfExperiments

        self.printPerformance(avePerformance)

    def printPerformance(self, avePerformance):
        print('Average training and test statistics:')
        print('\t\t\tTraining\tTest')
        for key in avePerformance.keys():
            temp = avePerformance.get(key)
            print(key + ': ' + "%.3f" % round(temp[0],3) + '\t' + "%.3f" % round(temp[1], 3))


class parallelRun():

    def __init__(self, majLP, minLP):
        self.defaultSplit = 0.7
        self.majLP = majLP
        self.minLP = minLP

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
                completeTrainFileName = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + ".txt")
                completeValidFileName = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + ".txt")
                trainDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-csv.csv")
                validDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-csv.csv")
                completeDataFileName = os.path.join(DATA_FOLDER, DATA_HEADER, DATA_HEADER + "-csv.csv")

                if os.path.isfile(completeTrainFileName):      # training.txt exists
                    pass
                else:
                    if os.path.isfile(trainDataCSV):           # training.csv exists
                        convertCSV2TXT(trainDataCSV, completeTrainFileName)
                        convertCSV2TXT(validDataCSV, completeValidFileName)
                    elif os.path.isfile(completeDataFileName):        # no data split exists, searching for complete.csv
                        completeData = pd.read_csv(completeDataFileName)
                        data_train, data_valid = train_test_split(completeData, test_size = 1 - self.defaultSplit,
                                                                     random_state = SEED_NUMBER)
                        data_train.to_csv(trainDataCSV)
                        data_valid.to_csv(validDataCSV)
                        convertCSV2TXT(os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-csv.csv"), completeTrainFileName)
                        convertCSV2TXT(os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-csv.csv"), completeValidFileName)
                dataManage = DataManage(completeTrainFileName, completeValidFileName)
                for it in range(NO_EXPERIMENTS_AVERAGING):
                    argument = []
                    argument.append(it + 1)
                    argument.append(dataManage)
                    argument.append(self.majLP)
                    argument.append(self.minLP)
                    arg_instances.append(argument)
                Parallel(n_jobs = NO_PARALLEL_JOBS, verbose = 1, backend = "threading")(map(delayed(MLRBC.MLRBC), arg_instances))
            else:
                completeTrainFileName = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + ".txt")
                completeValidFileName = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + ".txt")
                trainDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-csv.csv")
                validDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-csv.csv")
                completeDataFileName = os.path.join(DATA_FOLDER, DATA_HEADER, DATA_HEADER + "-csv.csv")

                if os.path.isfile(completeTrainFileName):      # training.txt exists
                    pass
                else:
                    if os.path.isfile(trainDataCSV):           # training.csv exists
                        convertCSV2TXT(trainDataCSV, completeTrainFileName)
                        convertCSV2TXT(validDataCSV, completeValidFileName)
                    elif os.path.isfile(completeDataFileName):        # no data split exists, searching for complete.csv
                        completeData = pd.read_csv(completeDataFileName)
                        data_train, data_valid = train_test_split(completeData, test_size = 1 - self.defaultSplit,
                                                                     random_state = SEED_NUMBER)
                        data_train.to_csv(trainDataCSV)
                        data_valid.to_csv(validDataCSV)
                        convertCSV2TXT(os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-csv.csv"), completeTrainFileName)
                        convertCSV2TXT(os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-csv.csv"), completeValidFileName)
                dataManage = DataManage(completeTrainFileName, completeValidFileName)
                MLRBC.MLRBC([1, dataManage, self.majLP, self.minLP])


def convertCSV2TXT(infilename, outfilename):
    """
    :param infileName: input .csv file name
    :param outfilename: output .txt file name
    """

    try:
        df = pd.read_csv(infilename)
        df.drop(df.columns[0], axis=1, inplace=True)
        Class = []
        for idx, row in df.iterrows():
            label = [int(l) for l in row[NO_ATTRIBUTES:]]
            newlabel = "".join(map(str, label))
            Class.append(newlabel)

        labelHeader = list(df.columns)
        dfCopy = df.copy()
        classHeader = labelHeader[NO_ATTRIBUTES:]
        dfCopy.drop(classHeader, axis=1, inplace = True)
        dfCopy["Class"] = Class

        data = dfCopy.values
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
    labelList = []
    classCount = {}
    for idx, row in df.iterrows():
        label = [int(l) for l in row[NO_ATTRIBUTES:]]
        newlabel = "".join(map(str, label))
        Class.append(newlabel)
        if newlabel in labelList:
            classCount[newlabel] += 1
        else:
            labelList.append(newlabel)
            classCount[newlabel] = 1

    print(str(len(classCount)) + " unique label powersets detected.")

    for key, value in classCount.items():
        if value == max(classCount.values()):
            majLP = key
        if value == min(classCount.values()):
            minLP = key
    print("Majority LP: " + majLP + " and Minority LP: " + minLP)
    lpIR = max(classCount.values()) / min(classCount.values())

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

    dataInfo = dict(zip(["card", "dens", "LP-IR", "MaxIR", "MeanIR", "IRLbls", "CVIR"], [card, dens, lpIR, maxIR, meanIR, IRLbls, CVIR]))
    print(dataInfo)
    return ([majLP, minLP])

def countLabel(label):
    count = 0
    for L in label:
        if float(L) != 0:
            count += 1
    return count

if __name__== "__main__":
    random.seed(SEED_NUMBER)
    completeDataFileName = os.path.join(DATA_FOLDER, DATA_HEADER, DATA_HEADER + "-csv.csv")
    [majLP, minLP] = dataProp(completeDataFileName)

    if NUMBER_OF_FOLDS < 2:
        numberOfExperiments = NO_EXPERIMENTS_AVERAGING
    else:
        numberOfExperiments = NUMBER_OF_FOLDS

    pathlib.Path(os.path.join(RUN_RESULT_PATH)).mkdir(parents=True, exist_ok=True)

    parallel = parallelRun(majLP, minLP)
    parallel.doParallel()

    if (DO_AVERAGING == True):
        averageTrack(numberOfExperiments)