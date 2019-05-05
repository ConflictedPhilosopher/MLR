import os

import random
import numpy as np
from config import *

class DataManage:
    def __init__(self, trainFile, testFile, classCount, dataInfo):

        random.seed(SEED_NUMBER)
        self.labelInstanceID = "InstanceID"  # Label for the data column header containing instance ID's.  If included label not found, algorithm assumes that no instance ID's were included.
        self.labelPhenotype = "Class"  # Label for the data column header containing the phenotype label. (Typically 'Class' for case/control datasets)
        self.labelMissingData = "NA"  # Label used for any missing data in the data set.
        self.discreteAttributeLimit = 10  # The maximum number of attribute states allowed before an attribute or phenotype is selfidered to be continuous (Set this value >= the number of states for any discrete attribute or phenotype in their dataset).
        self.discretePhenotypeLimit = 5000
        self.classCount = classCount
        self.dataInfo = dataInfo

        # Initialize global variables-------------------------------------------------
        self.numAttributes = None  # The number of attributes in the input file.
        self.areInstanceIDs = False  # Does the dataset contain a column of Instance IDs? (If so, it will not be included as an attribute)
        self.instanceIDRef = None  # The column reference for Instance IDs
        self.phenotypeRef = None  # The column reference for the Class/Phenotype column
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.MLphenotype = True
        self.attributeInfo = []  # Stores Discrete (0) or Continuous (1) for each attribute
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.ClassCount = 0

        # Train/Test Specific-----------------------------------------------------------------------------
        self.trainHeaderList = []  # The dataset column headers for the training data
        self.testHeaderList = []  # The dataset column headers for the testing data
        self.numTrainInstances = None  # The number of instances in the training data
        self.numTestInstances = None  # The number of instances in the testing data

        print("----------------------------------------------------------------------------")
        # Detect Features of training data--------------------------------------------------------------------------
        rawTrainData = self.loadData(trainFile, True)  # Load the raw data.

        if testFile == 'None':  # If no testing data is available, formatting relies solely on training data.
            data4Formating = rawTrainData
            # data4Formating = self.feasureSelection(data4Formating)
        else:
            rawTestData = self.loadData(testFile, False)
            self.numTestInstances = len(rawTestData)
            rawTrainData, rawTestData = self.feasureSelection(rawTrainData, rawTestData)
            data4Formating = rawTrainData + rawTestData  # Merge Training and Testing datasets

        self.characterizeDataset(rawTrainData)  # Detect number of attributes, instances, and reference locations.
        self.discriminatePhenotype(data4Formating)  # Determine if endpoint/phenotype is discrete or continuous.
        if self.discretePhenotype or self.MLphenotype:
            self.discriminateClasses(rawTrainData, rawTestData)  # Detect number of unique phenotype identifiers.
        # else:
        #     self.characterizePhenotype(data4Formating)

        self.discriminateAttributes(data4Formating)  # Detect whether attributes are discrete or continuous.
        self.characterizeAttributes(data4Formating)  # Determine potential attribute states or ranges.

        # Format and Shuffle Datasets----------------------------------------------------------------------------------------
        if testFile != 'None':
            self.testFormatted = self.formatData(rawTestData)  # Stores the formatted testing data set used throughout the algorithm.
            # self.testFormatted = self.sampleData(testFormatted, 0.3)
            self.numTestInstances = len(self.testFormatted)

        self.trainFormatted = self.formatData(rawTrainData)  # Stores the formatted training data set used throughout the algorithm.
        # self.trainFormatted = self.sampleData(trainFormatted, self.sampleSize)
        self.numTrainInstances = len(self.trainFormatted)
        print("----------------------------------------------------------------------------")

    def loadData(self, dataFile, doTrain):
        """ Load the data file. """
        # print("DataManagement: Loading Data... " + str(dataFile))
        datasetList = []
        try:
            f = open(dataFile, 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', dataFile)
            raise
        else:
            if doTrain:
                self.trainHeaderList = f.readline().rstrip('\n').split('\t')  # strip off first row
            else:
                self.testHeaderList = f.readline().rstrip('\n').split('\t')  # strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()
        return datasetList

    def feasureSelection(self, datasetTrain, datasetTest):
        featureRank = []
        datasetTrain = np.array(datasetTrain)
        datasetTest = np.array(datasetTest)

        try:
            f = open(os.path.join(DATA_FOLDER, DATA_HEADER, "feature-list.txt"), 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open' + "feature-list.txt")
            raise
        else:
            for line in f:
                lineList = line.strip('\n').split(' ')
                featureRank.append(lineList)
            f.close()

        featureCount = round(REDUCE_ATTRIBUTE * (len(datasetTrain[0])-1))
        featureList = [int(f[1])-1 for f in featureRank]
        selectionList = featureList[:featureCount] + [-1]
        reducedDatasetTrain = datasetTrain[:, selectionList].tolist()
        reducedDatasetTest = datasetTest[:, selectionList].tolist()

        self.trainHeaderList = self.trainHeaderList[0:len(selectionList)-1] + [self.trainHeaderList[-1]]
        self.testHeaderList = self.testHeaderList[0:len(selectionList)-1] + [self.testHeaderList[-1]]
        return (reducedDatasetTrain, reducedDatasetTest)

    def characterizeDataset(self, rawTrainData):
        # Detect Instance ID's and save location if they occur.  Then save number of attributes in data.
        column = 0
        if self.labelInstanceID in self.trainHeaderList:
            self.areInstanceIDs = True
            self.instanceIDRef = self.trainHeaderList.index(self.labelInstanceID)
            column =+ 1

        self.numAttributes = len(self.trainHeaderList) - column - 1

        # Identify location of phenotype column
        if self.labelPhenotype in self.trainHeaderList:
            self.phenotypeRef = self.trainHeaderList.index(self.labelPhenotype)
        else:
            print("DataManagement: Error - Phenotype column not found!  Check data set to ensure correct phenotype column label, or inclusion in the data.")

        # Adjust training header list to just include attributes labels

        if self.areInstanceIDs:
            if self.phenotypeRef > self.instanceIDRef:
                self.trainHeaderList.pop(self.phenotypeRef)
                self.trainHeaderList.pop(self.instanceIDRef)
            else:
                self.trainHeaderList.pop(self.instanceIDRef)
                self.trainHeaderList.pop(self.phenotypeRef)
        else:
            self.trainHeaderList.pop(self.phenotypeRef)

        # Store number of instances in training data
        self.numTrainInstances = len(rawTrainData)
        # print("DataManagement: Number of Attributes = " + str(self.numAttributes))
        print("DataManagement: Number of training instances = " + str(self.numTrainInstances))

    def discriminatePhenotype(self, rawData):
        """ Determine whether the phenotype is Discrete(class-based) or Continuous """
        # print("DataManagement: Analyzing Phenotype...")
        inst = 0
        classDict = {}
        while self.discretePhenotype and len(list(classDict.keys())) <= self.discretePhenotypeLimit and inst < (self.numTrainInstances + self.numTestInstances):  # Checks which discriminate between discrete and continuous attribute
            target = rawData[inst][self.phenotypeRef]
            if target in list(classDict.keys()):
                classDict[target] += 1
            elif target == self.labelMissingData:
                print("DataManagement: Warning - Individual detected with missing phenotype information!")
                pass
            else:  # New state observed
                classDict[target] = 1
            inst += 1
        ClassList = list(classDict.keys())
        if len(list(classDict.keys())) > self.discretePhenotypeLimit:
            self.discretePhenotype = False
            self.phenotypeList = [float(target), float(target)]
            # print("DataManagement: Phenotype Detected as Continuous.")
        elif len(ClassList[0]) > 1:
            self.ClassCount = len(ClassList[0])
            self.MLphenotype = True
            self.discretePhenotype = False
            # print("DataManagement: Multi-label phenotype Detected.")
        else:
            print("DataManagement: Phenotype Detected as Discrete.")

    def discriminateClasses(self, rawTrain, rawTest):
        """ Determines number of classes and their identifiers. Only used if phenotype is discrete. """
        # print("DataManagement: Detecting Classes...")
        inst = 0
        while inst < (len(rawTrain)):
            target = rawTrain[inst][self.phenotypeRef]
            if target in self.phenotypeList:
                pass
            else:
                self.phenotypeList.append(target)
            inst += 1
        print("Datamanagement: " + str(len(self.phenotypeList))+ " unique LPs detected for training data. ")
        temp = len(self.phenotypeList)

        for inst in range(len(rawTest)):
            target = rawTest[inst][self.phenotypeRef]
            if target in self.phenotypeList:
                pass
            else:
                self.phenotypeList.append(target)
        print("Datamanagement: " + str(len(self.phenotypeList) - temp) + " new unique LPs detected for test data.")
        # if self.MLphenotype:
            # print("Datamanagement: " + str(len(classCount)) + " unique label powersets are detected" )
            #for each in list(classCount.keys()):
                #print("Label Power set: " + str(each) + " count = " + str(classCount[each]))
        # else:
        #     print("DataManagement: Following Classes Detected:" + str(self.phenotypeList))
        #     for each in list(classCount.keys()):
        #         print("Class: " + str(each) + " count = " + str(classCount[each]))

    def compareDataset(self, rawTestData):
        " Ensures that the attributes in the testing data match those in the training data.  Also stores some information about the testing data. "
        if self.areInstanceIDs:
            if self.phenotypeRef > self.instanceIDRef:
                self.testHeaderList.pop(self.phenotypeRef)
                self.testHeaderList.pop(self.instanceIDRef)
            else:
                self.testHeaderList.pop(self.instanceIDRef)
                self.testHeaderList.pop(self.phenotypeRef)
        else:
            self.testHeaderList.pop(self.phenotypeRef)

        if self.trainHeaderList != self.testHeaderList:
            print("DataManagement: Error - Training and Testing Dataset Headers are not equivalent")

        # Stores the number of instances in the testing data.
        self.numTestInstances = len(rawTestData)
        # print("DataManagement: Number of Attributes = " + str(self.numAttributes))
        print("DataManagement: Number of test instances = " + str(self.numTestInstances))

    def discriminateAttributes(self, rawData):
        """ Determine whether attributes in dataset are discrete or continuous and saves this information. """
        # print("DataManagement: Detecting Attributes...")
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and instanceID columns)
                attIsDiscrete = True
                inst = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= self.discreteAttributeLimit and inst < self.numTrainInstances:  # Checks which discriminate between discrete and continuous attribute
                    target = rawData[inst][att]
                    if target in list(stateDict.keys()):  # Check if we've seen this attribute state yet.
                        stateDict[target] += 1
                    elif target == self.labelMissingData:  # Ignore missing data
                        pass
                    else:  # New state observed
                        stateDict[target] = 1
                    inst += 1

                if len(list(stateDict.keys())) > self.discreteAttributeLimit:
                    attIsDiscrete = False
                if attIsDiscrete:
                    self.attributeInfo.append([0, []])
                    self.discreteCount += 1
                else:
                    self.attributeInfo.append([1, [float(target), float(target)]])  # [min,max]
                    self.continuousCount += 1
        # print("DataManagement: Identified " + str(self.discreteCount) + " discrete and " + str(self.continuousCount) + " continuous attributes.")  # Debug

    def characterizeAttributes(self, rawData):
        """ Determine range (if continuous) or states (if discrete) for each attribute and saves this information"""
        # print("DataManagement: Characterizing Attributes...")
        attributeID = 0
        for att in range(len(rawData[0])):
            if att != self.instanceIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and instanceID columns)
                for inst in range(len(rawData)):
                    target = rawData[inst][att]
                    if not self.attributeInfo[attributeID][0]:  # If attribute is discrete
                        if target in self.attributeInfo[attributeID][1] or target == self.labelMissingData:
                            pass  # NOTE: Could potentially store state frequency information to guide learning.
                        else:
                            self.attributeInfo[attributeID][1].append(target)
                    else:  # If attribute is continuous

                        # Find Minimum and Maximum values for the continuous attribute so we know the range.
                        if target == self.labelMissingData:
                            pass
                        elif float(target) > self.attributeInfo[attributeID][1][1]:  # error
                            self.attributeInfo[attributeID][1][1] = float(target)
                        elif float(target) < self.attributeInfo[attributeID][1][0]:
                            self.attributeInfo[attributeID][1][0] = float(target)
                        else:
                            pass
                attributeID += 1

    def characterizePhenotype(self, rawData):
        """ Determine range of phenotype values. """
        # print("DataManagement: Characterizing Phenotype...")
        for inst in range(len(rawData)):
            target = rawData[inst][self.phenotypeRef]

            # Find Minimum and Maximum values for the continuous phenotype so we know the range.
            if target == self.labelMissingData:
                pass
            elif float(target) > self.phenotypeList[1]:
                self.phenotypeList[1] = float(target)
            elif float(target) < self.phenotypeList[0]:
                self.phenotypeList[0] = float(target)
            else:
                pass
        self.phenotypeRange = self.phenotypeList[1] - self.phenotypeList[0]

    def formatData(self, rawData):
        formatted = []
        for i in range(len(rawData)):
            formatted.append([None, None, None, None])  # [Attribute States, Phenotype, InstanceID]

        for inst in range(len(rawData)):
            stateList = []
            attributeID = 0
            for att in range(len(rawData[0])):
                if att != self.instanceIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and instanceID columns)
                    target = rawData[inst][att]

                    if self.attributeInfo[attributeID][0]:  # If the attribute is continuous
                        if target == self.labelMissingData:
                            stateList.append(target)  # Missing data saved as text label
                        else:
                            stateList.append(float(target))  # Save continuous data as floats.
                    else:  # If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                        stateList.append(target)  # missing data, and discrete variables, all stored as string objects
                    attributeID += 1

            # Final Format-----------------------------------------------
            formatted[inst][0] = stateList  # Attribute states stored here
            if self.discretePhenotype or self.MLphenotype:
                formatted[inst][1] = rawData[inst][self.phenotypeRef]  # phenotype stored here
            else:
                formatted[inst][1] = float(rawData[inst][self.phenotypeRef])
            if self.areInstanceIDs:
                formatted[inst][2] = rawData[inst][self.instanceIDRef]  # Instance ID stored here
            else:
                pass  # instance ID neither given nor required.
                # -----------------------------------------------------------
        random.shuffle(formatted)  # One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.
        return formatted

