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
from collections import Counter
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import networkx as nx
import seaborn as sns

from DataManagement import DataManage
import MLRBC
import RebootModel
from FPS_Clustering import density_based
from config import *


class averageTrack():

    def __init__(self, NumberOfExperiment):
        self.NumberOfExperiments = NumberOfExperiment

        self.aveTrack()
        self.saveAverage()
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
        h = [row[8] for row in datasetList]  # TN
        j = [row[9] for row in datasetList]  # over-General accuracy

        self.IterNum = a[0:int(len(a) / self.NumberOfExperiments)]
        macroPopSize = []
        self.macroPopSize_ave = []
        for i in range(self.NumberOfExperiments):
            macroPopSize.append(
                (b[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
        macroPopSize = np.sum(macroPopSize, axis=0)
        for x in macroPopSize:
            self.macroPopSize_ave.append(x / self.NumberOfExperiments)

        microPopSize = []
        self.microPopSize_ave = []
        for i in range(self.NumberOfExperiments):
            microPopSize.append(
                (c[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
        microPopSize = np.sum(microPopSize, axis=0)
        for x in microPopSize:
            self.microPopSize_ave.append(x / self.NumberOfExperiments)

        if PLOT_SETTING[0]:
            plt.figure(1)
            plt.plot(self.IterNum, self.macroPopSize_ave, 'r-', label='MacroPopSize')
            plt.plot(self.IterNum, self.microPopSize_ave, 'b-', label='MicroPopSize')
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

        accuracyEstimate = []
        self.accuracyEstimate_ave = []
        for i in range(self.NumberOfExperiments):
            accuracyEstimate.append(
                (e[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
        accuracyEstimate = np.sum(accuracyEstimate, axis=0)
        for x in accuracyEstimate:
            self.accuracyEstimate_ave.append(x / self.NumberOfExperiments)

        overGenAccuracy = []
        self.overGenAccuracy_ave = []
        for i in range(self.NumberOfExperiments):
            overGenAccuracy.append((j[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
        overGenAccuracy = np.sum(overGenAccuracy, axis=0)
        for x in overGenAccuracy:
            self.overGenAccuracy_ave.append(x / self.NumberOfExperiments)

        if PLOT_SETTING[1]:
            plt.figure(2)
            plt.plot(self.IterNum, self.accuracyEstimate_ave, 'b-', label = 'Accuracy Estimate')
            plt.plot(self.IterNum, self.overGenAccuracy_ave, 'r-', label = 'Over-general Accuracy Estimate')
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

        hlossEstimate = []
        self.hlossEstimate_ave = []
        for i in range(self.NumberOfExperiments):
            hlossEstimate.append(
                (d[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
        hlossEstimate = np.sum(hlossEstimate, axis=0)
        for x in hlossEstimate:
            self.hlossEstimate_ave.append(x / self.NumberOfExperiments)

        if PLOT_SETTING[2]:
            plt.figure(3)
            plt.plot(self.IterNum, self.hlossEstimate_ave, '-b', label = "Hamming loss")
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

        aveGenerality = []
        self.aveGenerality_ave = []
        for i in range(self.NumberOfExperiments):
            aveGenerality.append(
                (f[i * int(len(a) / self.NumberOfExperiments):(i + 1) * int(len(a) / self.NumberOfExperiments)]))
        aveGenerality = np.sum(aveGenerality, axis=0)
        for x in aveGenerality:
            self.aveGenerality_ave.append(x / self.NumberOfExperiments)

        if PLOT_SETTING[3]:
            plt.figure(4)
            plt.plot(self.IterNum, self.aveGenerality_ave, 'b-', label='AveGenerality')
            legend = plt.legend(loc='center', shadow=True, fontsize='large')
            plt.xlabel('Iteration')
            plt.ylim([0, 1])

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

        if PLOT_SETTING[4]:
            plt.figure(5)
            plt.plot(self.IterNum, self.tp_ave, 'b-', label='TP')
            plt.plot(self.IterNum, self.tn_ave, 'r-', label='TN')
            legend = plt.legend(loc='center', shadow=True, fontsize='small')
            plt.xlabel('Iteration')
            plt.ylim(bottom = 0.0)

        plt.show()

    def saveAverage(self):

        file_name = DATA_HEADER + "_MLRBC_AveTrack" + '.txt'
        completeName = os.path.join(RUN_RESULT_PATH, file_name)
        try:
            learnTrackOut = open(completeName, 'w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', completeName)
            raise
        else:
            learnTrackOut.write("Iteration\tMacroP\tMicroP\tHL\tAcc\tGen\ttp\ttn\tOverGenAcc\n")

        for i in range(len(self.IterNum)):
            trackString = str(self.IterNum[i]) + "\t" + str(self.macroPopSize_ave[i]) + "\t" + str(self.microPopSize_ave[i]) \
                          + "\t" + str("%.4f" % self.hlossEstimate_ave[i]) + "\t" + str("%.4f" % self.accuracyEstimate_ave[i]) + "\t" \
                          + str("%.4f" % self.aveGenerality_ave[i]) + "\t" + str("%.4f" % self.tp_ave[i]) + "\t" + str("%.4f" % self.tn_ave[i]) + "\t" + str("%.4f" % self.overGenAccuracy_ave[i]) + "\n"
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

    def __init__(self):
        self.defaultSplit = 0.7

    def doParallel(self):
        arg_instances = []
        if NUMBER_OF_FOLDS > 1:

            dataList = []
            for i in range(NUMBER_OF_FOLDS):
                foldCSV = os.path.join(DATA_FOLDER, DATA_HEADER, DATA_HEADER + "-" + str(i + 1) + ".csv")
                dataList.append(pd.read_csv(foldCSV))

            for i in range(NUMBER_OF_FOLDS):
                trainDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-" + str(i + 1) + "-csv.csv")
                validDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-" + str(i + 1) + "-csv.csv")
                frames = [dataList[j] for j in range(NUMBER_OF_FOLDS) if j != i]
                dfTrain = pd.concat(frames)
                dfValid = dataList[i]
                dfTrain.to_csv(trainDataCSV, index = False)
                dfValid.to_csv(validDataCSV, index = False)

            for it in range(NUMBER_OF_FOLDS):
                argument = []
                argument.append(it+1)
                completeTrainFileName = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-" + str(it + 1) + ".txt")
                completeValidFileName = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-" + str(it + 1) + ".txt")
                trainDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-" + str(it + 1) + "-csv.csv")
                validDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-" + str(it + 1) + "-csv.csv")
                if os.path.isfile(completeTrainFileName):      # training.txt exists
                    pass
                else:
                    if os.path.isfile(trainDataCSV):           # training.csv exists
                        convertCSV2TXT(trainDataCSV, completeTrainFileName)
                        convertCSV2TXT(validDataCSV, completeValidFileName)
                    else:
                        print("Error: Training/Validation Data not Found.")
                        break

                dataManage = DataManage(completeTrainFileName, completeValidFileName, self.classCount, self.dataInfo)
                argument.append(dataManage)
                argument.append(self.Card)
                argument.append(self.pi)
                argument.append(self.label_similarity)
                argument.append(self.label_clusters)
                if REBOOT_MODEL:
                    modelName = os.path.join(RUN_RESULT_PATH, DATA_HEADER + "_MLRBC_RulePop_" + str(it + 1) + ".txt")
                    Pop = RebootModel.ClassifierSet(dataManage, modelName)
                    argument.append(Pop)
                arg_instances.append(argument)
            measures = Parallel(n_jobs = NUMBER_OF_FOLDS, verbose=1, backend="multiprocessing")(map(delayed(MLRBC.MLRBC), arg_instances))
            self.meanPerformance(measures)
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
                dataManage = DataManage(completeTrainFileName, completeValidFileName, self.classCount, self.dataInfo)
                for it in range(NO_EXPERIMENTS_AVERAGING):
                    argument = []
                    argument.append(it)
                    argument.append(dataManage)
                    argument.append(self.Card)
                    argument.append(self.pi)
                    argument.append(self.label_similarity)
                    argument.append(self.label_clusters)
                    if REBOOT_MODEL:
                        modelName = os.path.join(RUN_RESULT_PATH,
                                                 DATA_HEADER + "_MLRBC_RulePop_" + str(it) + ".txt")
                        Pop = RebootModel.ClassifierSet(dataManage, modelName)
                        argument.append(Pop)
                    arg_instances.append(argument)
                measures = Parallel(n_jobs=NO_PARALLEL_JOBS, verbose=1, backend="multiprocessing")(map(delayed(MLRBC.MLRBC), arg_instances))
                # for meas in measures:
                #     print([round(val, 4) for val in meas.values()])
                self.meanPerformance(measures)
            else:
                completeTrainFileName = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + ".txt")
                completeValidFileName = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + ".txt")
                trainDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + ".csv")
                validDataCSV = os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + ".csv")
                completeDataFileName = os.path.join(DATA_FOLDER, DATA_HEADER, DATA_HEADER + ".csv")

                if os.path.isfile(completeTrainFileName):      # training.txt exists
                    pass
                else:
                    if os.path.isfile(trainDataCSV):           # training.csv exists
                        convertCSV2TXT(trainDataCSV, completeTrainFileName)
                        convertCSV2TXT(validDataCSV, completeValidFileName)
                    elif os.path.isfile(completeDataFileName):        # no data split exists, searching for complete.csv
                        completeData = pd.read_csv(completeDataFileName)
                        # completeDataSampled = self.tuneCard(completeData)
                        data_train, data_valid = train_test_split(completeData, test_size = 1 - self.defaultSplit,
                                                                     random_state = SEED_NUMBER)
                        data_train.to_csv(trainDataCSV)
                        data_valid.to_csv(validDataCSV)
                        convertCSV2TXT(os.path.join(DATA_FOLDER, DATA_HEADER, TRAIN_DATA_HEADER + "-csv.csv"), completeTrainFileName)
                        convertCSV2TXT(os.path.join(DATA_FOLDER, DATA_HEADER, VALID_DATA_HEADER + "-csv.csv"), completeValidFileName)

                dataManage = DataManage(completeTrainFileName, completeValidFileName, self.classCount, self.dataInfo)
                argument = []
                argument.append(1)
                argument.append(dataManage)
                argument.append(self.Card)
                argument.append(self.pi)
                argument.append(self.label_similarity)
                argument.append(self.label_clusters)
                if REBOOT_MODEL:
                    modelName = os.path.join(RUN_RESULT_PATH,
                                             DATA_HEADER + "_MLRBC_RulePop_" + str(1) + ".txt")
                    Pop = RebootModel.ClassifierSet(dataManage, modelName)
                    argument.append(Pop)
                measures = MLRBC.MLRBC(argument)
                # print([round(val, 4) for val in measures.values()])

    def meanPerformance(self, measures):
        """
        :param perfReports: multi-label performance measures for multiple experiments
        """
        perfReports = [run['perf'] for run in measures]
        att_track = [run['track'] for run in measures]
        total = sum(map(Counter, perfReports), Counter())
        meanPerf = {key: val/len(perfReports) for key, val in total.items()}
        print("Average ML performance:")
        print([round(val, 4) for val in meanPerf.values()])

        meanTrack = {}
        for track in att_track:
            for lp, value in track.items():
                if lp in meanTrack.keys():
                    meanTrack[lp] = [cur+v for (cur, v) in zip(meanTrack[lp], value)]
                else:
                    meanTrack[lp] = value
        meanTrack = {key: [v/len(att_track) for v in val] for key, val in meanTrack.items()}
        print("Average attribute tracking:")
        for key, value in meanTrack.items():
            print(key)
            print([round(val, 4) for val in value])

    def dataProp(self, infilename):
        """
        :param infilename: the complete dataset file in .csv format
        """
        try:
            df = pd.read_csv(infilename)
            Class = []  # list of all targets
            labels = []
            labelList = []
            classDict = {}
            for idx, row in df.iterrows():
                label = [int(l) for l in row[NO_ATTRIBUTES:]]
                labels.append(label)
                newlabel = "".join(map(str, label))
                Class.append(newlabel)
                if newlabel in labelList:
                    classDict[newlabel] += 1
                else:
                    labelList.append(newlabel)
                    classDict[newlabel] = 1

            print("dataProp: " + str(len(classDict)) + " unique LPs detected.")

            self.majLP = ""
            self.minLP = ""
            for key, value in classDict.items():
                if value == max(classDict.values()):
                    self.majLP = key
                if value == min(classDict.values()):
                    self.minLP = key
            lpIR = max(classDict.values()) / min(classDict.values())
            self.classCount = classDict

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
                count += self.countLabel(label)
            self.Card = count / dataCount
            dens = self.Card / classCount

            Y = np.empty([classCount])
            self.pi = np.empty([classCount])
            for y in range(classCount):
                sampleCount = 0
                for rowIdx, row in data.iterrows():
                    label = row["Class"]
                    if label[y] == '1':
                        sampleCount += 1
                Y[y] = sampleCount
            self.pi = Y / dataCount

            IRLbl = np.empty([classCount])
            maxIR = Y.max()
            for it in range(classCount):
                IRLbl[it] = (maxIR/Y[it])
            meanIR = IRLbl.sum() / classCount

            temp = (IRLbl - meanIR)**2
            IRLbls = sqrt(temp.sum() / (classCount - 1))
            CVIR = IRLbls / meanIR

            self.dataInfo = dict(zip(["card", "dens", "LP-IR", "MeanIR", "IRLbls", "CVIR"], [self.Card, dens, lpIR, meanIR, IRLbls, CVIR]))
            print("dataProp: " + str(self.dataInfo))

            self.label_similarity = self.similarity(labels, 'cosine')
            if CLUSTERING_METHOD == 'graph':
                self.label_clusters = self.graph(labels, self.label_similarity)
            elif CLUSTERING_METHOD == 'density':
                num_clusters = 2
                self.label_clusters = density_based(num_clusters, labels, 1 - self.label_similarity)
            else:
                print('Label clustering method not recognized!')
                return

        except FileNotFoundError:
            print("completeData.csv not found.")

    def similarity(self, label_matrix, measure):
        """
        :param labels: the complete set of labels
        :param measure: similarity measure to be calculated. 'co-occur', 'hamming', 'cosine'
        :return Sim: similarity based on hamming distance
        :return cosine: cosine similarity
        :return occurrence: Co-occurrence similarity
        """

        label_count = len(label_matrix[0])
        sim_measure = np.zeros((label_count, label_count))

        if measure == 'co-occur':
            for i in range(label_count):
                for j in range(label_count):
                    first_label = [l[i] for l in label_matrix]
                    second_label = [l[j] for l in label_matrix]
                    sim_measure[i, j] = np.dot(first_label, second_label) / np.linalg.norm(second_label, 1)
        else:
            if measure == 'hamming':
                for i in range(label_count):
                    first_label = [l[i] for l in label_matrix]
                    for j in range(i + 1, label_count):
                        second_label = [l[j] for l in label_matrix]
                        sim_measure[i, j] = np.sum(
                            np.array([1 for (l1, l2) in zip(first_label, second_label) if l1 == l2])) / len(labels)
                        sim_measure[j, i] = sim_measure[i, j]
            elif measure == 'cosine':
                label_matrix_sparse = sparse.csr_matrix(np.array(label_matrix).transpose())
                sim_measure = cosine_similarity(label_matrix_sparse)
                label_sim = open(os.path.join(RUN_RESULT_PATH, DATA_HEADER + "_label_sim.txt"), 'w')
                for row in sim_measure:
                    label_sim.write(" ".join([str(r) for r in row]) + "\n")
                label_sim.close()
                ax = sns.heatmap(
                    np.array(sim_measure),
                    vmin=0, vmax=1, center=0.5,
                    cmap=sns.diverging_palette(20, 220, n=200),
                    square=True
                )
                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=45,
                    horizontalalignment='right'
                )
                plt.show()
        return sim_measure

    def graph(self, labels, W):
        """
        :param labels:  the complete set of labels
        :return:
        """
        labels = np.array(labels)
        label_count = labels.shape[1]

        G = nx.Graph()
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        # e, v = np.linalg.eig(L)

        n_cluster = 2
        sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100, assign_labels = 'discretize',
                                random_state=SEED_NUMBER)
        sc.fit_predict(W)
        label_clusters = {}
        for n in range(int(n_cluster)):
            label_clusters[n] = [node for node in range(label_count) if sc.labels_[node] == n]

        edge_list = []
        for c1 in range(label_count):
            for c2 in range(c1+1, label_count):
                edge_exists = np.dot(labels[:, c1], labels[:, c2]) > 0
                if edge_exists:
                    edge_list.append((c1, c2))
                    w = W[c1, c2]
                    G.add_weighted_edges_from([(c1, c2, w)])
                else:
                    G.add_node(c1)
                    G.add_node(c2)

        fig1, ax1 = plt.subplots()
        ax1.set_title('Original data graph')
        pos = nx.spring_layout(G)
        for k in label_clusters.keys():
            nx.draw_networkx_nodes(G, pos,
                                   node_color=np.random.rand(3,),
                                   nodelist=label_clusters[k],
                                   )

        lbls = dict(zip(np.arange(label_count), [str(l) for l in np.arange(label_count)]))
        nx.draw_networkx_labels(G, pos, lbls, font_size=12)
        nx.draw_networkx_edges(G, pos, edge_list=edge_list, width=1, alpha=0.5)
        plt.savefig(DATA_HEADER + '_cosine_cluster')
        # plt.show()

        return label_clusters

    def card(self, data):
        """
        :return card: the multi-label class cardinality
        """
        count = 0.0
        for rowIdx, row in data.iterrows():
            label = row["Class"]
            count += self.countLabel(label)
        card = count / len(data)
        dens = card / len(label)

        return card

    def countLabel(self, label):
        count = 0
        for L in label:
            if float(L) != 0:
                count += 1
        return count


def convertCSV2TXT(infilename, outfilename):
    """
    :param infileName: input .csv file name
    :param outfilename: output .txt file name
    """

    try:
        df = pd.read_csv(infilename)
        # df.drop(df.columns[0], axis=1, inplace=True)
        if "Class" in df:
            dfCopy = df.astype({"Class": str})
        else:
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


if __name__== "__main__":

    random.seed(SEED_NUMBER)
    parallel = parallelRun()
    print('Training MLR model on ' + DATA_HEADER + ' dataset.')

    completeDataFileName = os.path.join(DATA_FOLDER, DATA_HEADER, DATA_HEADER + ".csv")
    parallel.dataProp(completeDataFileName)

    pathlib.Path(os.path.join(RUN_RESULT_PATH)).mkdir(parents=True, exist_ok=True)
    parallel.doParallel()

    if (DO_AVERAGING == True):
        averageTrack(NO_EXPERIMENTS_AVERAGING)