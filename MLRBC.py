import random
import copy
import math
import time
import os.path
from joblib import Parallel, delayed

import numpy as np
from sklearn.metrics import roc_curve, auc

from config import *

def MLRBC(arg):
    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ###### Major Run Parameters - Essential to be set correctly for a successful run of the algorithm
    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    outFileName = DATA_HEADER							# Path/NewName for new algorithm output files. Note: Do not give a file extension, this is done automatically.
    learningIterations = str(NO_TRAIN_ITERATION)		# Specify complete algorithm evaluation checkpoints and maximum number of learning iterations (e.g. 1000.2000.5000 = A maximum of 5000 learning iterations with evaluations at 1000, 2000, and 5000 iterations)
    N = POP_SIZE									    # Maximum size of the rule population (a.k.a. Micro-classifier population size, where N is the sum of the classifier numerosities in the population)
    p_spec = 1 - P_HASH 									    # The probability of specifying an attribute when covering. (1-p_spec = the probability of adding '#' in ternary rule representations). Greater numbers of attributes in a dataset will require lower values of p_spec.
    exp = arg[0]
    random.seed(exp)

    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ###### Logistical Run Parameters
    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    labelInstanceID = "InstanceID"					# Label for the data column header containing instance ID's.  If included label not found, algorithm assumes that no instance ID's were included.
    labelPhenotype = "Class"							# Label for the data column header containing the phenotype label. (Typically 'Class' for case/control datasets)
    labelMissingData = "NA"							# Label used for any missing data in the data set.
    discreteAttributeLimit = "NA"					# The maximum number of attribute states allowed before an attribute or phenotype is considered to be continuous (Set this value >= the number of states for any discrete attribute or phenotype in their dataset).
    discretePhenotypeLimit = "NA"
    trackingFrequency = 5000 #NO_TRAIN_ITERATION						# Specifies the number of iterations before each estimated learning progress report by the algorithm ('0' = report progress every epoch, i.e. every pass through all instances in the training data).

    ######----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ###### Supervised Learning Parameters - Generally just use default values.
    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    nu = 10									# (v) Power parameter used to determine the importance of high accuracy when calculating fitness. (typically set to 5, recommended setting of 1 in noisy data)
    chi = 0.8										# (X) The probability of applying crossover in the GA. (typically set to 0.5-1.0)
    upsilon = 0.04									# (u) The probability of mutating an allele within an offspring.(typically set to 0.01-0.05)
    theta_GA = 50									# The GA threshold; The GA is applied in a set when the average time since the last GA in the set is greater than theta_GA.
    theta_del = 20									# The deletion experience threshold; The calculation of the deletion probability changes once this threshold is passed.
    theta_sub = 200									# The subsumption experience threshold;
    acc_sub = 0.99									# Subsumption accuracy requirement
    beta = 0.1										# Learning parameter; Used in calculating average correct set size
    delta = 0.1										# Deletion parameter; Used in determining deletion vote calculation.
    init_fit = 0.01									# The initial fitness for a new classifier. (typically very small, approaching but not equal to zero)
    fitnessReduction = 0.1							# Initial fitness reduction in GA offspring rules.
    error = 0.000001
    ######-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ###### Algorithm Heuristic Options
    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    doSubsumption = 0								# Activate Subsumption? (1 is True, 0 is False).  Subsumption is a heuristic that actively seeks to increase generalization in the rule population.
    selectionMethod = 'tournament'		    		# Select GA parent selection strategy ('tournament' or 'roulette')
    theta_sel = 0.2									# The fraction of the correct set to be included in tournament selection.

    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ###### PopulationReboot - An option to begin LCS learning from an existing, saved rule population. Note that the training data is re-shuffled during a reboot.
    ######--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    doPopulationReboot = REBOOT_MODEL							# Start eLCS from an existing rule population? (1 is True, 0 is False).
    popRebootPath = "None"


    class Timer:
        def __init__(self):
            # Global Time objects
            self.globalStartRef = time.time()
            self.globalTime = 0.0
            self.addedTime = 0.0

            # Match Time Variables
            self.startRefMatching = 0.0
            self.globalMatching = 0.0

            # Deletion Time Variables
            self.startRefDeletion = 0.0
            self.globalDeletion = 0.0

            # Subsumption Time Variables
            self.startRefSubsumption = 0.0
            self.globalSubsumption = 0.0

            # Selection Time Variables
            self.startRefSelection = 0.0
            self.globalSelection = 0.0

            # Evaluation Time Variables
            self.startRefEvaluation = 0.0
            self.globalEvaluation = 0.0


            # ************************************************************

        def startTimeMatching(self):
            """ Tracks MatchSet Time """
            self.startRefMatching = time.time()

        def stopTimeMatching(self):
            """ Tracks MatchSet Time """
            diff = time.time() - self.startRefMatching
            self.globalMatching += diff

            # ************************************************************

        def startTimeDeletion(self):
            """ Tracks Deletion Time """
            self.startRefDeletion = time.time()

        def stopTimeDeletion(self):
            """ Tracks Deletion Time """
            diff = time.time() - self.startRefDeletion
            self.globalDeletion += diff

        # ************************************************************
        def startTimeSubsumption(self):
            """Tracks Subsumption Time """
            self.startRefSubsumption = time.time()

        def stopTimeSubsumption(self):
            """Tracks Subsumption Time """
            diff = time.time() - self.startRefSubsumption
            self.globalSubsumption += diff

            # ************************************************************

        def startTimeSelection(self):
            """ Tracks Selection Time """
            self.startRefSelection = time.time()

        def stopTimeSelection(self):
            """ Tracks Selection Time """
            diff = time.time() - self.startRefSelection
            self.globalSelection += diff

        # ************************************************************
        def startTimeEvaluation(self):
            """ Tracks Evaluation Time """
            self.startRefEvaluation = time.time()

        def stopTimeEvaluation(self):
            """ Tracks Evaluation Time """
            diff = time.time() - self.startRefEvaluation
            self.globalEvaluation += diff

            # ************************************************************

        def returnGlobalTimer(self):
            """ Set the global end timer, call at very end of algorithm. """
            self.globalTime = (
                              time.time() - self.globalStartRef) + self.addedTime  # Reports time in minutes, addedTime is for population reboot.
            return self.globalTime / 60.0

        def setTimerRestart(self, remakeFile):
            """ Sets all time values to the those previously evolved in the loaded popFile.  """
            try:
                fileObject = open(remakeFile + "_PopStats.txt", 'r')  # opens each datafile to read.
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', remakeFile + "_PopStats.txt")
                raise

            timeDataRef = 18

            tempLine = None
            for i in range(timeDataRef):
                tempLine = fileObject.readline()
            tempList = tempLine.strip().split('\t')
            self.addedTime = float(tempList[1]) * 60  # previous global time added with Reboot.

            tempLine = fileObject.readline()
            tempList = tempLine.strip().split('\t')
            self.globalMatching = float(tempList[1]) * 60

            tempLine = fileObject.readline()
            tempList = tempLine.strip().split('\t')
            self.globalDeletion = float(tempList[1]) * 60

            tempLine = fileObject.readline()
            tempList = tempLine.strip().split('\t')
            self.globalSubsumption = float(tempList[1]) * 60

            tempLine = fileObject.readline()
            tempList = tempLine.strip().split('\t')
            self.globalSelection = float(tempList[1]) * 60

            tempLine = fileObject.readline()
            tempList = tempLine.strip().split('\t')
            self.globalEvaluation = float(tempList[1]) * 60

            fileObject.close()

        ##############################################################################################

        def reportTimes(self):
            """ Reports the time summaries for this run. Returns a string ready to be printed out."""
            outputTime = "Global Time\t" + str(self.globalTime / 60.0) + \
                         "\nMatching Time\t" + str(self.globalMatching / 60.0) + \
                         "\nDeletion Time\t" + str(self.globalDeletion / 60.0) + \
                         "\nSubsumption Time\t" + str(self.globalSubsumption / 60.0) + \
                         "\nSelection Time\t" + str(self.globalSelection / 60.0) + \
                         "\nEvaluation Time\t" + str(self.globalEvaluation / 60.0) + "\n"

            return outputTime


    class Constants:
        def setConstants(self):
            """ Takes the parameters parsed as a dictionary from eLCS_ConfigParser and saves them as global constants. """

            dataFormatted = arg[1]
            # Major Run Parameters -----------------------------------------------------------------------------------------
            self.trainFile = dataFormatted.trainFormatted  # par['trainFile']                                       #Saved as text
            self.testFile = dataFormatted.testFormatted  # par['testFile']                                         #Saved as text
            self.outFileName = outFileName + '_MLRBC'  # str(par['outFileName'])+'_eLCS'                  #Saved as text
            self.learningIterations = learningIterations  # par['learningIterations']                     #Saved as text
            self.N = N  # int(par['N'])                                                  #Saved as integer
            self.p_spec = p_spec  # float(par['p_spec'])                                      #Saved as float
            self.majLP = arg[2]
            self.minLP = arg[3]
            self.error = error

            # Logistical Run Parameters ------------------------------------------------------------------------------------
            # if randomSeed == False:
            #     self.useSeed = False  # Saved as Boolean
            # else:
            #     self.useSeed = True  # Saved as Boolean
            #     self.randomSeed = randomSeed  # int(par['randomSeed'])                #Saved as integer

            self.labelInstanceID = labelInstanceID  # par['labelInstanceID']                           #Saved as text
            self.labelPhenotype = labelPhenotype  # par['labelPhenotype']                             #Saved as text
            self.labelMissingData = labelMissingData  # par['labelMissingData']                         #Saved as text
            self.discreteAttributeLimit = discreteAttributeLimit  # int(par['discreteAttributeLimit'])        #Saved as integer
            self.trackingFrequency = trackingFrequency  # int(par['trackingFrequency'])                  #Saved as integer
            self.discretePhenotypeLimit = discretePhenotypeLimit

            # Supervised Learning Parameters -------------------------------------------------------------------------------
            self.nu = nu  # int(par['nu'])                                                #Saved as integer
            self.chi = chi  # float(par['chi'])                                            #Saved as float
            self.upsilon = upsilon  # float(par['upsilon'])                                    #Saved as float
            self.theta_GA = theta_GA  # int(par['theta_GA'])                                    #Saved as integer
            self.theta_del = theta_del  # int(par['theta_del'])                                  #Saved as integer
            self.theta_sub = theta_sub  # int(par['theta_sub'])                                  #Saved as integer
            self.acc_sub = acc_sub  # float(par['acc_sub'])                                    #Saved as float
            self.beta = beta  # float(par['beta'])                                          #Saved as float
            self.delta = delta  # float(par['delta'])
            self.init_fit = init_fit  # float(par['init_fit'])                                  #Saved as float
            self.fitnessReduction = fitnessReduction  # float(par['fitnessReduction'])
            self.ruleCompactionMethod = None #'A9'

            # Algorithm Heuristic Options --New-------------------------------------------------------------------------
            self.doSubsumption = doSubsumption  # bool(int(par['doSubsumption']))                    #Saved as Boolean
            self.selectionMethod = selectionMethod  # par['selectionMethod']                           #Saved as text
            self.theta_sel = theta_sel  # float(par['theta_sel'])                                #Saved as float

            # PopulationReboot -------------------------------------------------------------------------------
            self.doPopulationReboot = doPopulationReboot  # bool(int(par['doPopulationReboot']))          #Saved as Boolean
            self.popRebootPath = popRebootPath  # par['popRebootPath']                               #Saved as text

        # New
        def referenceTimer(self, timer):
            """ Store reference to the timer object. """
            self.timer = timer

        def referenceEnv(self, e):
            """ Store reference to environment object. """
            self.env = e

        def parseIterations(self):
            """ Parse the 'learningIterations' string to identify the maximum number of learning iterations as well as evaluation checkpoints. """
            checkpoints = self.learningIterations.split('.')

            for i in range(len(checkpoints)):
                checkpoints[i] = int(checkpoints[i])

            self.learningCheckpoints = checkpoints  # next two lines needed for reboot
            self.maxLearningIterations = self.learningCheckpoints[(len(self.learningCheckpoints) - 1)]  # ???

            if self.trackingFrequency == 0:
                self.trackingFrequency = self.env.formatData.numTrainInstances  # Adjust tracking frequency to match the training data size - learning tracking occurs once every epoch


    #To access one of the above constant values from another module, import GHCS_Constants * and use "cons.something"
    cons = Constants()
    cons.setConstants() #Store run parameters in the 'Constants' module.
    cons.parseIterations() #Store run parameters in the 'Constants' module.


    class Offline_Environment:
        def __init__(self):
            # Initialize global variables-------------------------------------------------
            self.dataRef = 0
            self.storeDataRef = 0
            self.formatData = arg[1]

            # Initialize the first dataset instance to be passed to eLCS
            self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
            self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][1]
            if cons.testFile == 'None':
                pass
            else:
                self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
                self.currentTestPhenotype = self.formatData.testFormatted[self.dataRef][1]

        def getTrainInstance(self):
            """ Returns the current training instance. """
            return [self.currentTrainState, self.currentTrainPhenotype]

        def getTestInstance(self):
            """ Returns the current training instance. """
            return [self.currentTestState, self.currentTestPhenotype]

        def newInstance(self, isTraining):
            """  Shifts the environment to the next instance in the data. """
            # -------------------------------------------------------
            # Training Data
            # -------------------------------------------------------
            if isTraining:
                if self.dataRef < (self.formatData.numTrainInstances - 1):
                    self.dataRef += 1
                    self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
                    self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][1]
                else:  # Once learning has completed an epoch (i.e. a cycle of iterations though the entire training dataset) it starts back at the first instance in the data)
                    self.resetDataRef(isTraining)

            # -------------------------------------------------------
            # Testing Data
            # -------------------------------------------------------
            else:
                if self.dataRef < (self.formatData.numTestInstances - 1):
                    self.dataRef += 1
                    self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
                    self.currentTestPhenotype = self.formatData.testFormatted[self.dataRef][1]

        def resetDataRef(self, isTraining):
            """ Resets the environment back to the first instance in the current data set. """
            self.dataRef = 0
            if isTraining:
                self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
                self.currentTrainPhenotype = self.formatData.trainFormatted[self.dataRef][1]
            else:
                self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
                self.currentTestPhenotype = self.formatData.testFormatted[self.dataRef][1]

        def startEvaluationMode(self):
            """ Turns on evaluation mode.  Saves the instance we left off in the training data. """
            self.storeDataRef = self.dataRef

        def stopEvaluationMode(self):
            """ Turns off evaluation mode.  Re-establishes place in dataset."""
            self.dataRef = self.storeDataRef


    class Classifier:
        def __init__(self, a=None, b=None, c=None, d=None):
            # Major Parameters --------------------------------------------------
            self.specifiedAttList = []  # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
            self.condition = []  # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
            self.phenotype = None  # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous

            self.fitness = cons.init_fit  # Classifier fitness - initialized to a constant initial fitness value
            self.accuracy = 0.0  # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
            self.numerosity = 1  # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
            self.aveMatchSetSize = None  # A parameter used in deletion which reflects the size of match sets within this rule has been included.
            self.deletionVote = None  # The current deletion weight for this classifier.

            # Experience Management ---------------------------------------------
            self.timeStampGA = None  # Time since rule last in a correct set.
            self.initTimeStamp = None  # Iteration in which the rule first appeared.

            # Classifier Accuracy Tracking -------------------------------------
            self.matchCount = 0  # Known in many LCS implementations as experience, i.e. the total number of times this classifier was in a match set
            self.experience = {}
            self.correctCount = 0  # The total number of times this classifier was in a correct set
            self.loss = 0
            self.precision = 0
            self.recall = 0
            self.f1 = 0
            self.acc = 0

            if isinstance(c, list):
                self.classifierCovering(a, b, c, d)
            elif isinstance(a, Classifier):
                self.classifierCopy(a, b)
            elif isinstance(a, list) and b == None:
                self.rebootClassifier(a)
            else:
                print("Classifier: Error building classifier.")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # CLASSIFIER CONSTRUCTION METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def classifierCovering(self, setSize, exploreIter, state, phenotype):
            """ Makes a new classifier when the covering mechanism is triggered.  The new classifier will match the current training instance.
            Covering will NOT produce a default rule (i.e. a rule with a completely general condition). """
            # Initialize new classifier parameters----------
            self.timeStampGA = exploreIter
            self.initTimeStamp = exploreIter
            self.aveMatchSetSize = setSize
            dataInfo = cons.env.formatData
            # -------------------------------------------------------
            # DISCRETE PHENOTYPE
            # -------------------------------------------------------
            if dataInfo.discretePhenotype or dataInfo.MLphenotype:
                self.phenotype = phenotype
            # -------------------------------------------------------
            # CONTINUOUS PHENOTYPE
            # -------------------------------------------------------
            else:
                phenotypeRange = dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]
                rangeRadius = random.randint(25, 75) * 0.01 * phenotypeRange / 2.0  # Continuous initialization domain radius.
                Low = float(phenotype) - rangeRadius
                High = float(phenotype) + rangeRadius
                self.phenotype = [Low, High]  # ALKR Representation, Initialization centered around training instance  with a range between 25 and 75% of the domain size.
            # -------------------------------------------------------
            # GENERATE MATCHING CONDITION
            # -------------------------------------------------------
            # while len(self.specifiedAttList) < 1:
            for attRef in range(len(state)):
                if random.random() < cons.p_spec and state[attRef] != cons.labelMissingData:
                    self.specifiedAttList.append(attRef)
                    self.condition.append(self.buildMatch(attRef, state))
            # if self.isOverGeneral():
            #     zero = []
            #     for i in range(len(phenotype)):
            #         zero.append('0')
            #     zero = "".join(zero)
            #     self.phenotype = zero

        def classifierCopy(self, clOld, exploreIter):
            """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity
            is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate
            offspring based on parent classifiers."""
            self.specifiedAttList = copy.deepcopy(clOld.specifiedAttList)
            self.condition = copy.deepcopy(clOld.condition)
            self.phenotype = copy.deepcopy(clOld.phenotype)
            self.timeStampGA = exploreIter
            self.initTimeStamp = exploreIter
            self.aveMatchSetSize = copy.deepcopy(clOld.aveMatchSetSize)
            self.fitness = clOld.fitness
            self.accuracy = clOld.accuracy

        def rebootClassifier(self, classifierList):
            """ Rebuilds a saved classifier as part of the population Reboot """
            numAttributes = cons.env.formatData.numAttributes
            attInfo = cons.env.formatData.attributeInfo
            for attRef in range(0, numAttributes):
                if classifierList[attRef] != '#':  # Attribute in rule is not wild
                    if attInfo[attRef][0]:  # Continuous Attribute
                        valueRange = classifierList[attRef].split(';')
                        self.condition.append(valueRange)
                        self.specifiedAttList.append(attRef)
                    else:
                        self.condition.append(classifierList[attRef])
                        self.specifiedAttList.append(attRef)
            # -------------------------------------------------------
            # DISCRETE PHENOTYPE
            # -------------------------------------------------------
            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                self.phenotype = str(classifierList[numAttributes])
            # -------------------------------------------------------
            # CONTINUOUS PHENOTYPE
            # -------------------------------------------------------
            else:
                self.phenotype = classifierList[numAttributes].split(';')
                for i in range(2):
                    self.phenotype[i] = float(self.phenotype[i])

            self.fitness = float(classifierList[numAttributes + 1])
            self.accuracy = float(classifierList[numAttributes + 2])
            self.numerosity = int(classifierList[numAttributes + 3])
            self.aveMatchSetSize = float(classifierList[numAttributes + 4])
            self.timeStampGA = int(classifierList[numAttributes + 5])
            self.initTimeStamp = int(classifierList[numAttributes + 6])
            self.deletionVote = float(classifierList[numAttributes + 8])
            self.correctCount = int(classifierList[numAttributes + 9])
            self.matchCount = int(classifierList[numAttributes + 10])

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # MATCHING
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def match(self, state):
            """ Returns if the classifier matches in the current situation. """
            for i in range(len(self.condition)):
                attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
                # -------------------------------------------------------
                # CONTINUOUS ATTRIBUTE
                # -------------------------------------------------------
                if attributeInfo[0]:
                    instanceValue = state[self.specifiedAttList[i]]
                    if self.condition[i][0] < instanceValue < self.condition[i][1] or instanceValue == cons.labelMissingData:
                        pass
                    else:
                        return False
                        # -------------------------------------------------------
                # DISCRETE ATTRIBUTE
                # -------------------------------------------------------
                else:
                    stateRep = state[self.specifiedAttList[i]]
                    if stateRep == self.condition[i] or stateRep == cons.labelMissingData:
                        pass
                    else:
                        return False
            return True

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # GENETIC ALGORITHM MECHANISMS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def uniformCrossover(self, cl):
            """ Applies uniform crossover and returns if the classifiers changed. Handles both discrete and continuous attributes.
            #SWARTZ: self. is where for the better attributes are more likely to be specified
            #DEVITO: cl. is where less useful attribute are more likely to be specified
            """
            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype or random.random() < 0.5:  # Always crossover condition if the phenotype is discrete (if continuous phenotype, half the time phenotype crossover is performed instead)
                p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList)
                p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList)

                # Make list of attribute references appearing in at least one of the parents.-----------------------------
                comboAttList = []
                for i in p_self_specifiedAttList:
                    comboAttList.append(i)
                for i in p_cl_specifiedAttList:
                    if i not in comboAttList:
                        comboAttList.append(i)
                    elif not cons.env.formatData.attributeInfo[i][0]:  # Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                        comboAttList.remove(i)
                comboAttList.sort()
                # --------------------------------------------------------------------------------------------------------
                changed = False
                for attRef in comboAttList:  # Each condition specifies different attributes, so we need to go through all attributes in the dataset.
                    attributeInfo = cons.env.formatData.attributeInfo[attRef]
                    probability = 0.5  # Equal probability for attribute alleles to be exchanged.
                    # -----------------------------
                    ref = 0
                    # if attRef in self.specifiedAttList:
                    if attRef in p_self_specifiedAttList:
                        ref += 1
                    # if attRef in cl.specifiedAttList:
                    if attRef in p_cl_specifiedAttList:
                        ref += 1
                    # -----------------------------

                    if ref == 0:  # Attribute not specified in either condition (Attribute type makes no difference)
                        print("Error: UniformCrossover!")
                        pass

                    elif ref == 1:  # Attribute specified in only one condition - do probabilistic switch of whole attribute state (Attribute type makes no difference)
                        if attRef in p_self_specifiedAttList and random.random() > probability:
                            i = self.specifiedAttList.index(attRef)  # reference to the position of the attribute in the rule representation
                            cl.condition.append(self.condition.pop(i))  # Take attribute from self and add to cl
                            cl.specifiedAttList.append(attRef)
                            self.specifiedAttList.remove(attRef)
                            changed = True  # Remove att from self and add to cl

                        if attRef in p_cl_specifiedAttList and random.random() < probability:
                            i = cl.specifiedAttList.index(
                                attRef)  # reference to the position of the attribute in the rule representation
                            self.condition.append(cl.condition.pop(i))  # Take attribute from self and add to cl
                            self.specifiedAttList.append(attRef)
                            cl.specifiedAttList.remove(attRef)
                            changed = True  # Remove att from cl and add to self.


                    else:  # Attribute specified in both conditions - do random crossover between state alleles.  The same attribute may be specified at different positions within either classifier
                        # -------------------------------------------------------
                        # CONTINUOUS ATTRIBUTE
                        # -------------------------------------------------------
                        if attributeInfo[0]:
                            i_cl1 = self.specifiedAttList.index(attRef)  # pairs with self (classifier 1)
                            i_cl2 = cl.specifiedAttList.index(attRef)  # pairs with cl (classifier 2)
                            tempKey = random.randint(0,3)  # Make random choice between 4 scenarios, Swap minimums, Swap maximums, Self absorbs cl, or cl absorbs self.
                            if tempKey == 0:  # Swap minimum
                                temp = self.condition[i_cl1][0]
                                self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                                cl.condition[i_cl2][0] = temp
                            elif tempKey == 1:  # Swap maximum
                                temp = self.condition[i_cl1][1]
                                self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                                cl.condition[i_cl2][1] = temp
                            else:  # absorb range
                                allList = self.condition[i_cl1] + cl.condition[i_cl2]
                                newMin = min(allList)
                                newMax = max(allList)
                                if tempKey == 2:  # self absorbs cl
                                    self.condition[i_cl1] = [newMin, newMax]
                                    # Remove cl
                                    cl.condition.pop(i_cl2)
                                    cl.specifiedAttList.remove(attRef)
                                else:  # cl absorbs self
                                    cl.condition[i_cl2] = [newMin, newMax]
                                    # Remove self
                                    self.condition.pop(i_cl1)
                                    self.specifiedAttList.remove(attRef)
                        # -------------------------------------------------------
                        # DISCRETE ATTRIBUTE
                        # -------------------------------------------------------
                        else:
                            pass

                tempList1 = copy.deepcopy(p_self_specifiedAttList)
                tempList2 = copy.deepcopy(cl.specifiedAttList)
                tempList1.sort()
                tempList2.sort()

                if changed and (tempList1 == tempList2):
                    changed = False

                return changed
            # -------------------------------------------------------
            # CONTINUOUS PHENOTYPE CROSSOVER
            # -------------------------------------------------------
            else:
                return self.phenotypeCrossover(cl)

        def phenotypeCrossover(self, cl):
            """ Crossover a continuous phenotype """
            changed = False
            if self.phenotype[0] == cl.phenotype[0] and self.phenotype[1] == cl.phenotype[1]:
                return changed
            else:
                tempKey = random.random() < 0.5  # Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
                if tempKey:  # Swap minimum
                    temp = self.phenotype[0]
                    self.phenotype[0] = cl.phenotype[0]
                    cl.phenotype[0] = temp
                    changed = True
                elif tempKey:  # Swap maximum
                    temp = self.phenotype[1]
                    self.phenotype[1] = cl.phenotype[1]
                    cl.phenotype[1] = temp
                    changed = True

            return changed

        def Mutation(self, state, phenotype):
            """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
            changed = False
            Acc = ClassAccuracy()
            # -------------------------------------------------------
            # MUTATE CONDITION
            # -------------------------------------------------------
            for attRef in range(cons.env.formatData.numAttributes):  # Each condition specifies different attributes, so we need to go through all attributes in the dataset.
                attributeInfo = cons.env.formatData.attributeInfo[attRef]
                if random.random() < cons.upsilon and state[attRef] != cons.labelMissingData:
                    # MUTATION--------------------------------------------------------------------------------------------------------------
                    if attRef not in self.specifiedAttList:  # Attribute not yet specified
                        self.specifiedAttList.append(attRef)
                        self.condition.append(self.buildMatch(attRef, state))  # buildMatch handles both discrete and continuous attributes
                        changed = True

                    elif attRef in self.specifiedAttList:  # Attribute already specified
                        i = self.specifiedAttList.index(attRef)  # reference to the position of the attribute in the rule representation
                        # -------------------------------------------------------
                        # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
                        # -------------------------------------------------------
                        # if not attributeInfo[0] or random.random() > 0.5:
                        if not attributeInfo[0] or random.random() > cons.p_spec:

                            self.specifiedAttList.remove(attRef)
                            self.condition.pop(i)  # buildMatch handles both discrete and continuous attributes
                            changed = True
                        # -------------------------------------------------------
                        # CONTINUOUS ATTRIBUTE - (mutate range with 50% probability vs. removing specification of this attribute all together)
                        # -------------------------------------------------------
                        else:
                            # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                            attRange = float(attributeInfo[1][1]) - float(attributeInfo[1][0])
                            mutateRange = random.random() * 0.5 * attRange
                            if random.random() > 0.5:  # Mutate minimum
                                if random.random() > 0.5:  # Add
                                    self.condition[i][0] += mutateRange
                                else:  # Subtract
                                    self.condition[i][0] -= mutateRange
                            else:  # Mutate maximum
                                if random.random() > 0.5:  # Add
                                    self.condition[i][1] += mutateRange
                                else:  # Subtract
                                    self.condition[i][1] -= mutateRange

                            # Repair range - such that min specified first, and max second.
                            self.condition[i].sort()
                            changed = True
                    # -------------------------------------------------------
                    # NO MUTATION OCCURS
                    # -------------------------------------------------------
                    else:
                        pass
            # -------------------------------------------------------
            # MUTATE PHENOTYPE
            # -------------------------------------------------------
            if cons.env.formatData.discretePhenotype:
                nowChanged = self.discretePhenotypeMutation()
            elif cons.env.formatData.MLphenotype:
                nowChanged = self.MLphenotypeMutation(phenotype)
                while (Acc.countLabel(self.phenotype) == 0):
                    nowChanged = self.MLphenotypeMutation(phenotype)
            else:
                nowChanged = self.continuousPhenotypeMutation(phenotype)

            if changed or nowChanged:
                return True

        def discretePhenotypeMutation(self):
            """ Mutate this rule's discrete phenotype. """
            changed = False
            if random.random() < cons.upsilon:
                phenotypeList = copy.deepcopy(cons.env.formatData.phenotypeList)
                phenotypeList.remove(self.phenotype)
                newPhenotype = random.sample(phenotypeList, 1)
                self.phenotype = newPhenotype[0]
                changed = True

            return changed

        def MLphenotypeMutation(self, phenotype):
            changed = False
            newPhenotype = list(self.phenotype)
            loc = 0
            for (L_hat, L) in zip(self.phenotype, phenotype):
                if random.random() < cons.upsilon:
                    if (float(L) == float(L_hat)):
                        pass
                    elif float(L_hat) == 0:
                        newPhenotype[loc] = '1'
                        changed = True
                    else:
                        newPhenotype[loc] = '0'
                        changed = True
                loc += 1

            self.phenotype = "".join(newPhenotype)
            return changed

        def continuousPhenotypeMutation(self, phenotype):
            """ Mutate this rule's continuous phenotype. """
            changed = False
            if random.random() < cons.upsilon:  # Mutate continuous phenotype
                phenRange = self.phenotype[1] - self.phenotype[0]
                mutateRange = random.random() * 0.5 * phenRange
                tempKey = random.randint(0,2)  # Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
                if tempKey == 0:  # Mutate minimum
                    if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                        self.phenotype[0] += mutateRange
                    else:  # Subtract
                        self.phenotype[0] -= mutateRange
                    changed = True
                elif tempKey == 1:  # Mutate maximum
                    if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                        self.phenotype[1] -= mutateRange
                    else:  # Subtract
                        self.phenotype[1] += mutateRange
                    changed = True
                else:  # mutate both
                    if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                        self.phenotype[0] += mutateRange
                    else:  # Subtract
                        self.phenotype[0] -= mutateRange
                    if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                        self.phenotype[1] -= mutateRange
                    else:  # Subtract
                        self.phenotype[1] += mutateRange
                    changed = True

                # Repair range - such that min specified first, and max second.
                self.phenotype.sort()
            # ---------------------------------------------------------------------
            return changed

            # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # SUBSUMPTION METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def subsumes(self, cl):
            """ Returns if the classifier (self) subsumes cl """
            # -------------------------------------------------------
            # DISCRETE PHENOTYPE
            # -------------------------------------------------------
            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                if cl.phenotype == self.phenotype:
                    if self.isSubsumer() and self.isMoreGeneral(cl):
                        return True
                return False
            # -------------------------------------------------------
            # CONTINUOUS PHENOTYPE -  NOTE: for continuous phenotypes, the subsumption intuition is reversed, i.e. While a subsuming rule condition is more general, a subsuming phenotype is more specific.
            # -------------------------------------------------------
            else:
                if self.phenotype[0] >= cl.phenotype[0] and self.phenotype[1] <= cl.phenotype[1]:
                    if self.isSubsumer() and self.isMoreGeneral(cl):
                        return True
                return False

        def isSubsumer(self):
            """ Returns if the classifier (self) is a possible subsumer. A classifier must be as or more accurate than the classifier it is trying to subsume.  """
            if self.matchCount > cons.theta_sub and self.accuracy > cons.acc_sub:
                return True
            return False

        def isMoreGeneral(self, cl):
            """ Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. """
            if len(self.specifiedAttList) >= len(cl.specifiedAttList):
                return False
            for i in range(len(self.specifiedAttList)):  # Check each attribute specified in self.condition
                attributeInfo = cons.env.formatData.attributeInfo[self.specifiedAttList[i]]
                if self.specifiedAttList[i] not in cl.specifiedAttList:
                    return False
                # -------------------------------------------------------
                # CONTINUOUS ATTRIBUTE
                # -------------------------------------------------------
                if attributeInfo[0]:
                    otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                    # If self has a narrower ranger of values than it is a subsumer
                    if self.condition[i][0] < cl.condition[otherRef][0]:
                        return False
                    if self.condition[i][1] > cl.condition[otherRef][1]:
                        return False

            return True

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # DELETION METHOD
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def getDelProp(self, meanFitness):
            """  Returns the vote for deletion of the classifier. """
            if self.fitness >= cons.delta * meanFitness or self.matchCount < cons.theta_del:
            # if self.fitness / self.numerosity >= cons.delta * meanFitness or self.matchCount < cons.theta_del:

                self.deletionVote = self.aveMatchSetSize * self.numerosity

            elif self.fitness == 0.0:
                self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (cons.init_fit / self.numerosity)
            else:
                # self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness)
                self.deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
            return self.deletionVote

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # OTHER METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def buildMatch(self, attRef, state):
            """ Builds a matching condition for the classifierCovering method. """
            attributeInfo = cons.env.formatData.attributeInfo[attRef]
            # -------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            # -------------------------------------------------------
            if attributeInfo[0]:
                attRange = attributeInfo[1][1] - attributeInfo[1][0]
                rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # Continuous initialization domain radius.
                Low = state[attRef] - rangeRadius
                High = state[attRef] + rangeRadius
                condList = [Low, High]  # ALKR Representation, Initialization centered around training instance  with a
                                        # range between 25% and 75% of the domain size.
            # -------------------------------------------------------
            # DISCRETE ATTRIBUTE
            # -------------------------------------------------------
            else:
                condList = state[attRef]  # State already formatted like GABIL in DataManagement

            return condList

        def equals(self, cl):
            """ Returns if the two classifiers are identical in condition and phenotype. This works for discrete or continuous attributes or phenotypes. """
            if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList):  # Is phenotype the same and are the same number of attributes specified - quick equality check first.
                clRefs = sorted(cl.specifiedAttList)
                selfRefs = sorted(self.specifiedAttList)
                if clRefs == selfRefs:
                    for i in range(len(cl.specifiedAttList)):
                        tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i])
                        if cl.condition[i] == self.condition[tempIndex]:
                            pass
                        else:
                            return False
                    return True
            return False

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # PARAMETER UPDATES
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def updateAccuracy(self):
            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                # self.accuracy = self.correctCount / float(self.matchCount)
                self.accuracy = 1 - (self.loss / float(self.matchCount))


        def updateFitness(self, matchSetNumAcc):
            if cons.env.formatData.MLphenotype:
                if self.matchCount < 1.0 / cons.beta:
                    self.fitness = pow(self.accuracy, cons.nu)
                else:
                    self.fitness = pow(self.accuracy, cons.nu)
                    # k = self.accuracy * self.numerosity /(matchSetNumAcc)
                    # self.fitness = self.fitness + cons.beta * (k - self.fitness)
            else:
                if (self.phenotype[1] - self.phenotype[0]) >= cons.env.formatData.phenotypeRange:
                    self.fitness = 0.0
                else:
                    self.fitness = math.fabs(pow(self.accuracy, cons.nu) - (self.phenotype[1] - self.phenotype[0]) / cons.env.formatData.phenotypeRange)

        def updateMatchSetSize(self, matchSetSize):
            """  Updates the average match set size. """
            if self.matchCount < 1.0 / cons.beta:
                self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount - 1) + matchSetSize) / float(self.matchCount)
            else:
                self.aveMatchSetSize = self.aveMatchSetSize + cons.beta * (matchSetSize - self.aveMatchSetSize)

        def updateExperience(self, TruePhenotype):
            """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
            if TruePhenotype in self.experience.keys():
                self.experience[TruePhenotype] += 1
            else:
                self.experience[TruePhenotype] = 1
            self.matchCount += 1

        def updateCorrect(self):
            """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
            self.correctCount += 1

        def updateLoss(self, TruePhenotype):
            # Update the loss value of the classifier
            a = np.array(list(TruePhenotype)).astype(int)
            b = np.array(list(self.phenotype)).astype(int)
            loss = sum([e for e in a ^ b]) / len(self.phenotype)
            self.loss += loss

        def updateMLperformance(self, TruePhenotype, combVote):  # New

            Acc = ClassAccuracy()
            Acc.multiLablePerformace(self.phenotype, TruePhenotype, combVote)
            self.precision += Acc.getPrecisionSingle()
            self.recall += Acc.getRecallSingle()
            self.f1 += Acc.getFmeasureSingle()
            self.loss += Acc.getLossSingle()
            self.acc += Acc.getAccuracySingle()

        def updateNumerosity(self, num):
            """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
            self.numerosity += num

        def updateTimeStamp(self, ts):
            """ Sets the time stamp of the classifier. """
            self.timeStampGA = ts

        def setAccuracy(self, acc):
            """ Sets the accuracy of the classifier """
            self.accuracy = acc

        def setFitness(self, fit):
            """  Sets the fitness of the classifier. """
            self.fitness = fit

        def reportClassifier(self):
            """  Transforms the rule representation used to a more standard readable format. """
            numAttributes = cons.env.formatData.numAttributes
            thisClassifier = []
            counter = 0
            for i in range(numAttributes):
                if i in self.specifiedAttList:
                    thisClassifier.append(self.condition[counter])
                    counter += 1
                else:
                    thisClassifier.append('#')
            return thisClassifier

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def printClassifier(self):
            """ Formats and returns an output string describing this classifier. """
            classifierString = ""
            for attRef in range(cons.env.formatData.numAttributes):
                attributeInfo = cons.env.formatData.attributeInfo[attRef]
                if attRef in self.specifiedAttList:  # If the attribute was specified in the rule
                    i = self.specifiedAttList.index(attRef)
                    # -------------------------------------------------------
                    # CONTINUOUS ATTRIBUTE
                    # -------------------------------------------------------
                    if attributeInfo[0]:
                        classifierString += str("%.3f" % self.condition[i][0]) + ';' + str("%.3f" % self.condition[i][1]) + "\t"
                    # -------------------------------------------------------
                    # DISCRETE ATTRIBUTE
                    # -------------------------------------------------------
                    else:
                        classifierString += str(self.condition[i]) + "\t"
                else:  # Attribute is wild.
                    classifierString += '#' + "\t"
            # -------------------------------------------------------------------------------
            specificity = len(self.condition) / float(cons.env.formatData.numAttributes)

            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                classifierString += str(self.phenotype) + "\t"
            else:
                classifierString += str(self.phenotype[0]) + ';' + str(self.phenotype[1]) + "\t"

            # print(self.deletionVote)    does this not occur until population is full???
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            classifierString += str("%.3f" % self.fitness) + "\t" + str("%.3f" % self.accuracy) + "\t" + str(
                "%d" % self.numerosity) + "\t" + str("%.2f" % self.aveMatchSetSize) + "\t" + str(
                "%d" % self.timeStampGA) + "\t" + str("%d" % self.initTimeStamp) + "\t" + str("%.2f" % specificity) + "\t"
            # ???classifierString += str("%.2f" %self.deletionVote)+"\t"+str("%d" %self.correctCount)+"\t"+str("%d" %self.matchCount)+"\n"
            classifierString += "\t" + str("%d" % self.correctCount) + "\t" + str("%d" % self.matchCount) + "\n"

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            return classifierString

        def isOverGeneral(self):
            if len(self.specifiedAttList) < 1:
                return True
            else:
                return False


    class ClassifierSet:
        def __init__(self, a=None):
            """ Overloaded initialization: Handles creation of a new population or a rebooted population (i.e. a previously saved population). """
            # Major Parameters
            self.popSet = []  # List of classifiers/rules
            self.matchSet = []  # List of references to rules in population that match
            self.correctSet = []  # List of references to rules in population that both match and specify correct phenotype
            self.microPopSize = 0  # Tracks the current micro population size
            self.labelPowerSetList = []

            # Evaluation Parameters-------------------------------
            self.aveGenerality = 0.0
            self.expRules = 0.0
            self.attributeSpecList = []
            self.attributeAccList = []
            self.avePhenotypeRange = 0.0
            self.theta_GA = cons.theta_GA
            self.aveNumerosity = 0.0
            self.majLP = cons.majLP   # majority Label Powerset
            self.minLP = cons.minLP   # minority Label Powerset
            self.genAveAcc = 0  # Average accuracy of the over-general classifiers

            # Set Constructors-------------------------------------
            if a == None:
                self.makePop()  # Initialize a new population
            elif isinstance(a, str):
                self.rebootPop(a)  # Initialize a population based on an existing saved rule population
            else:
                print("ClassifierSet: Error building population.")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # POPULATION CONSTRUCTOR METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def makePop(self):
            """ Initializes the rule population """
            self.popSet = []

        def rebootPop(self, remakeFile):
            """ Remakes a previously evolved population from a saved text file. """
            print("Rebooting the following population: " + str(remakeFile) + "_RulePop.txt")
            # *******************Initial file handling**********************************************************
            try:
                datasetList = []
                f = open(remakeFile + "_RulePop.txt", 'r')
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', remakeFile + "_RulePop.txt")
                raise
            else:
                self.headerList = f.readline().rstrip('\n').split('\t')  # strip off first row
                for line in f:
                    lineList = line.strip('\n').split('\t')
                    datasetList.append(lineList)
                f.close()

                # **************************************************************************************************
            for each in datasetList:
                cl = Classifier(each)
                self.popSet.append(cl)
                self.microPopSize += 1
            print("Rebooted Rule Population has " + str(len(self.popSet)) + " Macro Pop Size.")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # CLASSIFIER SET CONSTRUCTOR METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        def singleMatchThread(self, args):
            cl = args[0]
            if cl.match(args[2]):
                return (args[1])

        def makeMatchSet(self, state_phenotype, exploreIter):
            """ Constructs a match set from the population. Covering is initiated if the match set is empty or a rule with the current correct phenotype is absent. """
            state = state_phenotype[0]
            phenotype = int(state_phenotype[1], 2)
            doCovering = True  # Covering check: Twofold (1)checks that a match is present, and (2) that at least one match dictates the correct phenotype.
            setNumerositySum = 0  # new
            k = 2
            overGenCount = 0
            overGenAccSum = 0

            # -------------------------------------------------------
            # MATCHING
            # -------------------------------------------------------
            cons.timer.startTimeMatching()

            if len(self.popSet) > DISTRIBUTED_MATCHING_TH:
                argList = []
                for i in range(len(self.popSet)):
                    arg = [self.popSet[i], i, state]
                    argList.append(arg)
                match = Parallel(n_jobs=15, verbose=0, backend="multiprocessing")(map(delayed(self.singleMatchThread), argList))
                self.matchSet = [x for x in match if x != None]
                for i in self.matchSet:
                    setNumerositySum += self.popSet[i].numerosity
                    if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                        prediction_int = int(self.popSet[i].phenotype, 2)
                        if prediction_int == phenotype:  # Check for phenotype coverage
                            doCovering = False
                    else:  # Continuous phenotype
                        if float(self.popSet[i].phenotype[0]) <= float(phenotype) <= float(
                                self.popSet[i].phenotype[1]):  # Check for phenotype coverage
                            doCovering = False
            else:
                popSize = len(self.popSet)
                for i in range(popSize):
                    cl = self.popSet[i]  # One classifier at a time
                    # calculate the average accuracy and count of over-general classifiers
                    # if cl.isOverGeneral():
                    #     overGenCount += 1
                    #     overGenAccSum += cl.accuracy
                    # theta_GA adaptation algorithm
                    if ADAPT_THETA_GA:
                        if cl.isOverGeneral():
                            try:
                                IR = cl.experience[self.majLP] / (cl.experience[self.majLP] + cl.experience[self.minLP])
                                if IR < 2000 and cl.numerosity > self.aveNumerosity:
                                    self.theta_GA = k * (cl.experience[self.majLP] + cl.experience[self.minLP])/ cl.experience[self.minLP]
                            except KeyError:
                                pass
                    else:
                        pass
                    #-------------------------------------------------------------
                    if cl.match(state):
                        self.matchSet.append(i)
                        setNumerositySum += cl.numerosity
                        # Covering Check--------------------------------------------------------
                        if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                            prediction_int = int(cl.phenotype, 2)
                            if prediction_int == phenotype:  # Check for phenotype coverage
                                doCovering = False
                        else:
                            if float(cl.phenotype[0]) <= float(phenotype) <= float(cl.phenotype[1]):  # Check for phenotype coverage
                                doCovering = False
                try:
                    self.aveNumerosity = setNumerositySum / len(self.popSet)
                    # self.overGenAcc = overGenAccSum / overGenCount
                except ZeroDivisionError:
                    self.aveNumerosity = 1
                    # self.overGenAcc = 0
            cons.timer.stopTimeMatching()  # new


            # -------------------------------------------------------
            # COVERING
            # -------------------------------------------------------
            while doCovering:
                newCl = Classifier(setNumerositySum + 1, exploreIter, state, state_phenotype[1])
                self.addClassifierToPopulation(newCl, True)
                self.matchSet.append(len(self.popSet) - 1)  # Add covered classifier to matchset
                doCovering = False

        def makeCorrectSet(self, phenotype):
            phenotype_int = int(phenotype, 2)
            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                # -------------------------------------------------------
                # DISCRETE AND MULTI-LABEL PHENOTYPE
                # -------------------------------------------------------
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    prediction_int = int(self.popSet[ref].phenotype, 2)
                    if prediction_int == phenotype_int:
                        self.correctSet.append(ref)
                    #elif cons.env.formatData.MLphenotype:
                        #if self.isPhenotypeSubset(ref, phenotype):
                            #self.correctSet.append(ref)
                    # -------------------------------------------------------
                    # CONTINUOUS PHENOTYPE
                    # -------------------------------------------------------
            else:
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if float(phenotype) <= float(self.popSet[ref].phenotype[1]) and float(phenotype) >= float(self.popSet[ref].phenotype[0]):
                        self.correctSet.append(ref)

        def isPhenotypeSubset(self, ref, phenotype):
            isSusbet = True
            cl_phenotype = self.popSet[ref].phenotype
            for (L, L_cl) in zip(phenotype, cl_phenotype):
                if L == '0' and L_cl != "0":
                    isSusbet = False
                    break
            return isSusbet

        def makeEvalMatchSet(self, state):
            """ Constructs a match set for evaluation purposes which does not activate either covering or deletion. """
            popSize = len(self.popSet)
            for i in range(popSize):  # Go through the population
                cl = self.popSet[i]  # A single classifier
                if cl.match(state):  # Check for match
                    self.matchSet.append(i)  # Add classifier to match set

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # CLASSIFIER DELETION METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def deletion(self, exploreIter):
            """ Returns the population size back to the maximum set by the user by deleting rules. """
            cons.timer.startTimeDeletion()
            while self.microPopSize > cons.N:
                self.deleteFromPopulation()
            cons.timer.stopTimeDeletion()

        def deleteFromPopulation(self):
            """ Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection
            considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. """
            meanFitness = self.getPopFitnessSum() / float(self.microPopSize)

            # Calculate total wheel size------------------------------
            sumCl = 0.0
            voteList = []
            for cl in self.popSet:
                vote = cl.getDelProp(meanFitness)
                sumCl += vote
                voteList.append(vote)
            # --------------------------------------------------------
            choicePoint = sumCl * random.random()  # Determine the choice point

            newSum = 0.0
            for i in range(len(voteList)):
                cl = self.popSet[i]
                newSum = newSum + voteList[i]
                if newSum > choicePoint:  # Select classifier for deletion
                    # Delete classifier----------------------------------
                    cl.updateNumerosity(-1)
                    self.microPopSize -= 1
                    if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                        self.removeMacroClassifier(i)
                        self.deleteFromMatchSet(i)
                        self.deleteFromCorrectSet(i)
                    return

            print("ClassifierSet: No eligible rules found for deletion in deleteFromPopulation.")
            return

        def removeMacroClassifier(self, ref):
            """ Removes the specified (macro-) classifier from the population. """
            self.popSet.pop(ref)

        def deleteFromMatchSet(self, deleteRef):
            """ Delete reference to classifier in population, contained in self.matchSet."""
            if deleteRef in self.matchSet:
                self.matchSet.remove(deleteRef)

            # Update match set reference list--------
            for j in range(len(self.matchSet)):
                ref = self.matchSet[j]
                if ref > deleteRef:
                    self.matchSet[j] -= 1

        def deleteFromCorrectSet(self, deleteRef):
            """ Delete reference to classifier in population, contained in self.corectSet."""
            if deleteRef in self.correctSet:
                self.correctSet.remove(deleteRef)

            # Update match set reference list--------
            for j in range(len(self.correctSet)):
                ref = self.correctSet[j]
                if ref > deleteRef:
                    self.correctSet[j] -= 1

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # GENETIC ALGORITHM
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def runGA(self, exploreIter, state, phenotype):
            """ The genetic discovery mechanism in eLCS is controlled here. """
            # -------------------------------------------------------
            # GA RUN REQUIREMENT
            # -------------------------------------------------------
            if (exploreIter - self.getIterStampAverage()) < self.theta_GA:  # Does the correct set meet the requirements for activating the GA?
                return

            self.setIterStamps(exploreIter)  # Updates the iteration time stamp for all rules in the correct set (which the GA opperates in).
            changed = False

            # -------------------------------------------------------
            # SELECT PARENTS - Niche GA - selects parents from the correct class
            # -------------------------------------------------------
            cons.timer.startTimeSelection()
            if cons.selectionMethod == "roulette":
                selectList = self.selectClassifierRW()
                clP1 = selectList[0]
                clP2 = selectList[1]
            elif cons.selectionMethod == "tournament":
                selectList = self.selectClassifierT()
                clP1 = selectList[0]
                clP2 = selectList[1]
            else:
                print("ClassifierSet: Error - requested GA selection method not available.")
            cons.timer.stopTimeSelection()

            # -------------------------------------------------------
            # INITIALIZE OFFSPRING
            # -------------------------------------------------------
            cl1 = Classifier(clP1, exploreIter)
            if clP2 == None:
                cl2 = Classifier(clP1, exploreIter)
            else:
                cl2 = Classifier(clP2, exploreIter)
            # -------------------------------------------------------
            # CROSSOVER OPERATOR - Uniform Crossover Implemented (i.e. all attributes have equal probability of crossing over between two parents)
            # -------------------------------------------------------
            if not cl1.equals(cl2) and random.random() < cons.chi:
                changed = cl1.uniformCrossover(cl2)

                # -------------------------------------------------------
            # INITIALIZE KEY OFFSPRING PARAMETERS
            # -------------------------------------------------------
            if changed:
                cl1.setAccuracy((cl1.accuracy + cl2.accuracy) / 2.0)
                cl1.setFitness(cons.fitnessReduction * (cl1.fitness + cl2.fitness) / 2.0)
                cl2.setAccuracy(cl1.accuracy)
                cl2.setFitness(cl1.fitness)
            else:
                cl1.setFitness(cons.fitnessReduction * cl1.fitness)
                cl2.setFitness(cons.fitnessReduction * cl2.fitness)
            # -------------------------------------------------------
            # MUTATION OPERATOR
            # -------------------------------------------------------
            nowchanged = cl1.Mutation(state, phenotype)
            howaboutnow = cl2.Mutation(state, phenotype)

            # -------------------------------------------------------
            # ADD OFFSPRING TO POPULATION
            # -------------------------------------------------------
            if changed or nowchanged or howaboutnow:
                self.insertDiscoveredClassifiers(cl1, cl2, clP1, clP2, exploreIter)  # Subsumption

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # SELECTION METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def selectClassifierRW(self):
            """ Selects parents using roulette wheel selection according to the fitness of the classifiers. """
            selectList = [None, None]
            setList = copy.deepcopy(self.correctSet)
            if len(setList) < 2:
                currentCount = 1  # Pick only one parent
            else:
                currentCount = 0  # Pick two parents
            # -----------------------------------------------
            # -----------------------------------------------
            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)

                choiceP = random.random() * fitSum
                i = 0
                ref = setList[i]
                sumCl = self.popSet[ref].fitness
                while choiceP > sumCl:
                    i = i + 1
                    ref = setList[i]  # WnB must be added to increment fitness
                    sumCl += self.popSet[ref].fitness

                selectList[currentCount] = self.popSet[ref]
                setList.remove(ref)
                currentCount += 1

            # -----------------------------------------------
            if selectList[0] == None:
                selectList[0] = selectList[1]

            return selectList

        def selectClassifierT(self):
            """  Selects parents using tournament selection according to the fitness of the classifiers. """
            selectList = [None, None]
            currentCount = 0
            setList = self.correctSet  # correct set is a list of reference IDs

            while currentCount < 2:
                tSize = int(len(setList) * cons.theta_sel)
                posList = random.sample(setList, tSize)

                bestF = 0
                bestC = self.correctSet[0]
                for j in posList:
                    if self.popSet[j].fitness > bestF:
                        bestF = self.popSet[j].fitness
                        bestC = j

                selectList[currentCount] = self.popSet[bestC]
                currentCount += 1

            return selectList


            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # SUBSUMPTION METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def subsumeClassifier(self, cl=None, cl1P=None, cl2P=None):
            """ Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set. """
            if cl1P != None and cl1P.subsumes(cl):
                self.microPopSize += 1
                cl1P.updateNumerosity(1)
            elif cl2P != None and cl2P.subsumes(cl):
                self.microPopSize += 1
                cl2P.updateNumerosity(1)
            else:
                self.subsumeClassifier2(cl)  # Try to subsume in the correct set.

        def subsumeClassifier2(self, cl):
            """ Tries to subsume a classifier in the correct set. If no subsumption is possible the classifier is simply added to the population considering
            the possibility that there exists an identical classifier. """
            choices = []
            for ref in self.correctSet:
                if self.popSet[ref].subsumes(cl):
                    choices.append(ref)

            if len(choices) > 0:  # Randomly pick one classifier to be subsumer
                choice = int(random.random() * len(choices))
                self.popSet[choices[choice]].updateNumerosity(1)
                self.microPopSize += 1
                return

            self.addClassifierToPopulation(cl,
                                           False)  # If no subsumer was found, check for identical classifier, if not then add the classifier to the population

        def doCorrectSetSubsumption(self):
            """ Executes correct set subsumption.  The correct set subsumption looks for the most general subsumer classifier in the correct set
            and subsumes all classifiers that are more specific than the selected one. """
            subsumer = None
            for ref in self.correctSet:
                cl = self.popSet[ref]
                if cl.isSubsumer():
                    if subsumer == None or cl.isMoreGeneral(subsumer):
                        subsumer = cl

            if subsumer != None:  # If a subsumer was found, subsume all more specific classifiers in the correct set
                i = 0
                while i < len(self.correctSet):
                    ref = self.correctSet[i]
                    if subsumer.isMoreGeneral(self.popSet[ref]):
                        subsumer.updateNumerosity(self.popSet[ref].numerosity)
                        self.removeMacroClassifier(ref)
                        self.deleteFromMatchSet(ref)
                        self.deleteFromCorrectSet(ref)
                        i = i - 1
                    i = i + 1

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # OTHER KEY METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def addClassifierToPopulation(self, cl, covering):
            """ Note new change here: Adds a classifier to the set and increases the microPopSize value accordingly."""
            oldCl = None
            if not covering:
                oldCl = self.getIdenticalClassifier(cl)
            if oldCl != None:  # found identical classifier
                oldCl.updateNumerosity(1)
                self.microPopSize += 1
            else:
                self.popSet.append(cl)
                self.microPopSize += 1

        def insertDiscoveredClassifiers(self, cl1, cl2, clP1, clP2, exploreIter):
            """ Inserts both discovered classifiers and activates GA subsumption if turned on. Also checks for default rule (i.e. rule with completely general condition) and
            prevents such rules from being added to the population, as it offers no predictive value within eLCS. """
            # -------------------------------------------------------
            # SUBSUMPTION
            # -------------------------------------------------------
            if cons.doSubsumption:
                cons.timer.startTimeSubsumption()

                if len(cl1.specifiedAttList) > 0:
                    self.subsumeClassifier(cl1, clP1, clP2)
                if len(cl2.specifiedAttList) > 0:
                    self.subsumeClassifier(cl2, clP1, clP2)

                cons.timer.stopTimeSubsumption()
            # -------------------------------------------------------
            # ADD OFFSPRING TO POPULATION
            # -------------------------------------------------------
            else:  # Just add the new classifiers to the population.
                if len(cl1.specifiedAttList) > 0:
                    self.addClassifierToPopulation(cl1,
                                                   False)  # False passed because this is not called for a covered rule.
                if len(cl2.specifiedAttList) > 0:
                    self.addClassifierToPopulation(cl2,
                                                   False)  # False passed because this is not called for a covered rule.

        # --Note New changes here-------------
        def updateSets(self, exploreIter, state_phenotype_conf):
            """ Updates all relevant parameters in the current match and correct sets. """

            matchSetNumerosity = 0
            matchSetNumAcc = 0
            for ref in self.matchSet:
                matchSetNumerosity += self.popSet[ref].numerosity

            for ref in self.matchSet:
                self.popSet[ref].updateExperience(state_phenotype_conf[1])
                self.popSet[ref].updateMatchSetSize(matchSetNumerosity)
                if ref in self.correctSet:
                    self.popSet[ref].updateCorrect()
                self.popSet[ref].updateLoss(state_phenotype_conf[1])
                # self.popSet[ref].updateMLperformance(state_phenotype_conf[1], None)  # New
                self.popSet[ref].updateAccuracy()
                self.popSet[ref].updateFitness(matchSetNumAcc)
                # matchSetNumAcc += self.popSet[ref].numerosity * self.popSet[ref].accuracy

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # OTHER METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def getIterStampAverage(self):
            """ Returns the average of the time stamps in the correct set. """
            sumCl = 0.0
            numSum = 0.0
            for i in range(len(self.correctSet)):
                ref = self.correctSet[i]
                sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
                numSum += self.popSet[ref].numerosity  # numerosity sum of correct set
                if numSum < 1:
                    print('Pause')
            return sumCl / float(numSum)

        def setIterStamps(self, exploreIter):
            """ Sets the time stamp of all classifiers in the set to the current time. The current time
            is the number of exploration steps executed so far.  """
            for i in range(len(self.correctSet)):
                ref = self.correctSet[i]
                self.popSet[ref].updateTimeStamp(exploreIter)

        def getFitnessSum(self, setList):
            """ Returns the sum of the fitnesses of all classifiers in the set. """
            sumCl = 0.0
            for i in range(len(setList)):
                ref = setList[i]
                sumCl += self.popSet[ref].fitness
            return sumCl

        def getPopFitnessSum(self):
            """ Returns the sum of the fitnesses of all classifiers in the set. """
            sumCl = 0.0
            for cl in self.popSet:
                sumCl += cl.fitness * cl.numerosity
            return sumCl

        def getIdenticalClassifier(self, newCl):
            """ Looks for an identical classifier in the population. """
            for cl in self.popSet:
                if newCl.equals(cl):
                    return cl
            return None

        def clearSets(self):
            """ Clears out references in the match and correct sets for the next learning iteration. """
            self.matchSet = []
            self.correctSet = []

        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # EVALUTATION METHODS
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        def runPopAveEval(self, exploreIter):
            """ Calculates some summary evaluations across the rule population including average generality. """
            genSum = 0
            for cl in self.popSet:
                genSum += ((cons.env.formatData.numAttributes - len(cl.condition)) / float(cons.env.formatData.numAttributes))

            if self.microPopSize == 0:
                self.aveGenerality = 'NA'
            else:
                self.aveGenerality = genSum / float(self.microPopSize)
                # -------------------------------------------------------
            # CONTINUOUS PHENOTYPE
            # -------------------------------------------------------
            if not (cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype):
                sumRuleRange = 0
                for cl in self.popSet:
                    sumRuleRange += (cl.phenotype[1] - cl.phenotype[0])
                phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
                self.avePhenotypeRange = (sumRuleRange / float(self.microPopSize)) / float(phenotypeRange)

        # New method
        def runAttGeneralitySum(self, isEvaluationSummary):
            """ Determine the population-wide frequency of attribute specification, and accuracy weighted specification.  Used in complete rule population evaluations. """
            self.attributeTracking = []
            if isEvaluationSummary:
                self.attributeSpecList = []
                self.attributeAccList = []
                for i in range(cons.env.formatData.numAttributes):
                    self.attributeSpecList.append(0)
                    self.attributeAccList.append(0.0)
                    self.attributeTracking.append(0)
                for cl in self.popSet:
                    for ref in cl.specifiedAttList:  # for each attRef
                        self.attributeTracking[ref] += 1
                        self.attributeSpecList[ref] += cl.numerosity
                        self.attributeAccList[ref] += cl.numerosity * cl.accuracy

            # print(self.attributeTracking)

        def getPopTrack(self, Hloss, accuracy, exploreIter):
            """ Returns a formated output string to be printed to the Learn Track output file. """
            OverGenAcc = 0.0
            TN = 0.0
            TP = 0.0
            trackString = str(exploreIter) + "\t" + str(len(self.popSet)) + "\t" + str(self.microPopSize) + "\t" \
                          + str(Hloss) + "\t" + str(accuracy) + "\t" + str("%.4f" % self.aveGenerality) \
                          + "\t" + str("%.3f" % TP) + "\t" + str("%.3f" % TN) + "\t" + str("%.3f" % OverGenAcc) \
                          + "\t" + str("%.2f" % cons.timer.returnGlobalTimer())  + "\n"
            """
            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:  # discrete phenotype
                print(("Epoch: " + str(int(exploreIter / trackingFrequency)) + "\t Iteration: " + str(
                    exploreIter) + "\t MacroPop: " + str(len(self.popSet)) + "\t MicroPop: " + str(
                    self.microPopSize) + "\t AveGen: " + str("%.2f" % self.aveGenerality) + "\t Time: " + str("%.2f" % cons.timer.returnGlobalTimer())))

       
            else:  # continuous phenotype
                print(("Epoch: " + str(int(exploreIter / trackingFrequency)) + "\t Iteration: " + str(
                    exploreIter) + "\t MacroPop: " + str(len(self.popSet)) + "\t MicroPop: " + str(
                    self.microPopSize) + "\t AccEstimate: " + str(accuracy) + "\t AveGen: " + str(
                    "%.2f" % self.aveGenerality) + "\t PhenRange: " + str(self.avePhenotypeRange) + "\t Time: " + str(
                    "%.2f" % cons.timer.returnGlobalTimer())))"""

            return trackString


    class Prediction:
        def __init__(self, population):
            self.decision = None
            self.population = population

        def calMaxPrediction(self):
            # -------------------------------------------------------
            # DISCRETE PHENOTYPES (CLASSES)
            # -------------------------------------------------------
            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:

                self.vote = {}
                self.tieBreak_Numerosity = {}
                self.tieBreak_TimeStamp = {}
                self.matchCount = {}

                # zero = [0.0] * len(cons.env.formatData.phenotypeList[0])

                phenotypeList = []

                for ref in self.population.matchSet:
                    cl = self.population.popSet[ref]
                    if any(cl.phenotype in s for s in self.population.labelPowerSetList):
                        pass
                    else:
                        self.population.labelPowerSetList.append(cl.phenotype)

                    if any(cl.phenotype in s for s in phenotypeList):
                        self.vote[cl.phenotype] += cl.fitness * cl.numerosity
                        self.tieBreak_Numerosity[cl.phenotype] += cl.numerosity
                        self.tieBreak_TimeStamp[cl.phenotype] += cl.initTimeStamp
                        self.matchCount[cl.phenotype] += 1
                    else:
                        phenotypeList.append(cl.phenotype)
                        self.vote[cl.phenotype] = cl.fitness * cl.numerosity
                        self.tieBreak_Numerosity[cl.phenotype] = cl.numerosity
                        self.tieBreak_TimeStamp[cl.phenotype] = cl.initTimeStamp
                        self.matchCount[cl.phenotype] = 1

                highVal = 0.0
                bestClass = []  # Prediction is set up to handle best class ties for problems with more than 2 classes
                for thisClass in phenotypeList:     #cons.env.formatData.phenotypeList:
                    if self.vote[thisClass] >= highVal:
                        highVal = self.vote[thisClass]

                for thisClass in phenotypeList:  #cons.env.formatData.phenotypeList:
                    if self.vote[thisClass] == highVal:  # Tie for best class
                        bestClass.append(thisClass)
                # ---------------------------
                if highVal == 0.0:
                    self.decision = None
                # -----------------------------------------------------------------------
                elif len(bestClass) > 1:  # Randomly choose between the best tied classes
                    bestNum = 0
                    newBestClass = []
                    for thisClass in bestClass:
                        if self.tieBreak_Numerosity[thisClass] >= bestNum:
                            bestNum = self.tieBreak_Numerosity[thisClass]

                    for thisClass in bestClass:
                        if self.tieBreak_Numerosity[thisClass] == bestNum:
                            newBestClass.append(thisClass)
                    # -----------------------------------------------------------------------
                    if len(newBestClass) > 1:  # still a tie
                        bestStamp = 0
                        newestBestClass = []
                        for thisClass in newBestClass:
                            if self.tieBreak_TimeStamp[thisClass] >= bestStamp:
                                bestStamp = self.tieBreak_TimeStamp[thisClass]

                        for thisClass in newBestClass:
                            if self.tieBreak_TimeStamp[thisClass] == bestStamp:
                                newestBestClass.append(thisClass)
                        # -----------------------------------------------------------------------
                        if len(newestBestClass) > 1:  # Prediction is completely tied - eLCS has no useful information for making a prediction
                            self.decision = 'Tie'
                        else:
                            self.decision = newestBestClass[0]
                    else:
                        self.decision = newBestClass[0]
                # ----------------------------------------------------------------------
                else:  # One best class determined by fitness vote
                    self.decision = bestClass[0]

            # -------------------------------------------------------
            # CONTINUOUS PHENOTYPES
            # -------------------------------------------------------
            else:
                if len(population.matchSet) < 1:
                    print("empty matchSet")
                    self.decision = None
                else:
                    # IDEA - outputs a single continuous prediction value(closeness to this prediction accuracy will dictate accuracy). In determining this value we examine all ranges
                    phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]  # Difference between max and min phenotype values observed in data.
                    predictionValue = 0
                    valueWeightSum = 0
                    for ref in population.matchSet:
                        cl = population.popSet[ref]
                        localRange = cl.phenotype[1] - cl.phenotype[0]
                        valueWeight = (phenotypeRange / float(localRange))
                        localAverage = cl.phenotype[1] + cl.phenotype[0] / 2.0

                        valueWeightSum += valueWeight
                        predictionValue += valueWeight * localAverage
                    if valueWeightSum == 0.0:
                        self.decision = None
                    else:
                        self.decision = predictionValue / float(valueWeightSum)

        def combinePredictions(self):

            self.combPred = None
            self.combVote = []

            for i in range(cons.env.formatData.ClassCount):
                self.combVote.append(0.0)

            if len(self.population.matchSet) == 0:
                self.combPred = None
            else:
                for ref in self.population.matchSet:
                    cl = self.population.popSet[ref]
                    if any(cl.phenotype in s for s in self.population.labelPowerSetList):
                        pass
                    else:
                        self.population.labelPowerSetList.append(cl.phenotype)

                    vote = cl.fitness * cl.numerosity
                    # vote = cl.fitness
                    # vote = cl.fitness * (cons.env.formatData.numAttributes - len(cl.specifiedAttList))/cons.env.formatData.numAttributes
                    it = 0
                    for l in cl.phenotype:
                        if (l == '1'):
                            self.combVote[it] += vote
                        it += 1
                maxVote = max(self.combVote)
                if maxVote > 0.0:
                    for i in range(len(self.combVote)):
                        self.combVote[i] /= maxVote

        def oneThreshold(self):
            pred = []
            loc = 0
            for val in self.combVote:
                if val >= THETA_THRESHOLD:
                    pred.append('1')
                else:
                    pred.append('0')
                loc += 1
            self.combPred = "".join(pred)

        def rcut(self):
            card = arg[4]
            pred = []
            if RCUT_T is None:
                t = round(card)
            else:
                t = RCUT_T

            labelRanks = [i[0] for i in sorted(enumerate(self.combVote), key=lambda x: x[1])]
            labelsetSelected = [l for l in labelRanks[cons.env.formatData.ClassCount - t:]]
            for i in range(cons.env.formatData.ClassCount):
                if i in labelsetSelected:
                    pred.append('1')
                else:
                    pred.append('0')
            self.combPred = "".join(pred)

        def pcut(self, voteList, numSamples):
            pi = arg[5]
            combPred = np.zeros((numSamples, cons.env.formatData.ClassCount), dtype=int)
            for col in range(cons.env.formatData.ClassCount):
                zeroPred = np.zeros(numSamples)
                P = voteList[:, col]
                sortIndex = np.argsort(P)
                k = pi[col] * numSamples * PCUT_W
                selectedSamples = sortIndex[-int(k):]
                for i in selectedSamples:
                    zeroPred[i] = 1
                combPred[:, col] = zeroPred
            combPred = combPred.astype(dtype=str)
            self.combPred = ["".join(row) for row in combPred]

        def getCombPred(self):
            return self.combPred

        def getCombVote(self):
            return self.combVote

        def getFitnessSum(self, population, low, high):
            """ Get the fitness sum of rules in the rule-set. For continuous phenotype prediction. """
            fitSum = 0
            for ref in population.matchSet:
                cl = population.popSet[ref]
                if cl.phenotype[0] <= low and cl.phenotype[1] >= high:  # if classifier range subsumes segment range.
                    fitSum += cl.fitness
            return fitSum

        def getDecision(self):
            return self.decision


    class ClassAccuracy:
        def __init__(self):
            classAccDict = dict(TP=0.0, FP=0.0, TN=0.0, FN=0.0)
            self.classAccDictList = [classAccDict] * cons.env.formatData.ClassCount

        def updateClassBased(self, target, combPred):
            """ Increment the appropriate cell of the confusion matrix """
            for it in range(cons.env.formatData.ClassCount):
                dictC = self.classAccDictList[it].copy()
                dictC["TP"] += int(target[it]) and int(combPred[it])
                dictC["FP"] += (int(target[it]) ^ int(combPred[it])) and int(combPred[it])
                dictC["TN"] += not (int(target[it]) or int(combPred[it]))
                dictC["FN"] += (int(target[it]) ^ int(combPred[it])) and int(target[it])
                self.classAccDictList[it] = dictC

        def microAverage(self):
            e = 0.001
            tpSum = 0
            tnSum = 0
            fpSum = 0
            fnSum = 0
            for l in range(cons.env.formatData.ClassCount):
                dictC = self.classAccDictList[l].copy()
                tpSum += dictC["TP"]
                tnSum += dictC["TN"]
                fpSum += dictC["FP"]
                fnSum += dictC["FN"]
            self.micro_precision = tpSum / (tpSum + fpSum + 1)
            self.micro_recall = tpSum / (tpSum + fnSum + 1)
            self.micro_f = 2 * (self.micro_precision * self.micro_recall) / (self.micro_precision + self.micro_recall + e)

        def macroAverage(self):
            e = 0.001
            self.macro_precision = 0.0
            self.macro_recall = 0.0
            for l in range(cons.env.formatData.ClassCount):
                dictC = self.classAccDictList[l].copy()
                self.macro_precision += (dictC["TP"] / (dictC["TP"] + dictC["FP"] + 1))
                self.macro_recall += (dictC["TP"] / (dictC["TP"] + dictC["FN"] + 1))
            self.macro_precision /= cons.env.formatData.ClassCount
            self.macro_recall /= cons.env.formatData.ClassCount
            self.macro_f = 2 * (self.macro_precision * self.macro_recall) / (self.macro_precision + self.macro_recall + e)


        def ExHammingLoss(self, phenotypePrediction, true_phenotype, phenotype_conf, true_conf):
            """
            Calculates the extended Hamming loss value for the given prediction according to 'Nazmi, Razegh-Jahromi,
            Homaifar, Multi-label classification  with weighted labels using learning classifiers, ICMLA-2017.'
            """
            it = 0
            Dist = 0.0
            if cons.env.formatData.areConfidence:
                for (L, C) in zip(phenotypePrediction, phenotype_conf):
                    Dist += math.pow((math.pow(float(L) - float(true_phenotype[it]), 2) + math.pow(C - true_conf[it], 2)), 0.5)
                    it += 1
            else:
                for L in phenotypePrediction:
                    Dist += math.fabs((float(L) - float(true_phenotype[it])))
                    it += 1
            loss = Dist/len(phenotypePrediction)
            return loss

        def labelCount(self, phenotypePrediction, true_phenotype):
            self.trueLabelCount = self.countLabel(true_phenotype)
            self.predictedLabelCount = self.countLabel(phenotypePrediction)

        def countLabel(self, phenotype):
            count = 0
            for L in phenotype:
                if float(L) != 0:
                    count += 1
            return count

        def intersection(self, phenotypePrediction, true_phenotype):
            self.Intersect = 0
            for (L, L_hat) in zip(phenotypePrediction, true_phenotype):
                if float(L) != 0:
                    self.Intersect += float(L)*float(L_hat)

        def union(self, phenotypePrediction, true_phenotype):
            self.Union = 0
            for (L, L_hat) in zip(phenotypePrediction, true_phenotype):
                if (float(L) or float(L_hat)) != 0:
                    self.Union += 1

        def precision(self):
            if self.trueLabelCount == 0:
                precision = 0.0
            else:
                precision = self.Intersect / self.trueLabelCount
            return precision

        def recall(self):
            if self.predictedLabelCount == 0:
                recall = 0.0
            else:
                recall = self.Intersect / self.predictedLabelCount
            return recall

        def accuracy(self):
            if self.Union == 0:
                accuracy = 0
            else:
                accuracy = self.Intersect / self.Union
            return accuracy

        def f_measure(self):
            if (self.trueLabelCount + self.predictedLabelCount) == 0:
                f_measure = 0
            else:
                f_measure = 2*self.Intersect/(self.trueLabelCount + self.predictedLabelCount)
            return f_measure

        def hammingLoss(self, phenotypePrediction, true_phenotype):
            Dist = 0.0
            for (L, L_hat) in zip(phenotypePrediction, true_phenotype):
                Dist += math.fabs((float(L) - float(L_hat)))
            loss = Dist/len(phenotypePrediction)
            return loss

        def oneError(self, combinedVote, true_phenotype):
            maxVote = max(combinedVote)
            maxIndex = []
            for (ind, v) in enumerate(combinedVote):
                if v == maxVote:
                    maxIndex.append(ind)
            error = 1
            for ind in maxIndex:
                if true_phenotype[ind] == '1':
                    error = 0
                    break
            return error

        def rankLoss(self, combinedVote, true_phenotype):
            L = []
            Lbar = []
            for (i, l) in enumerate(true_phenotype):
                if l == '0':
                    Lbar.append(i)
                else:
                    L.append(i)
            lossCount = 0
            for l in L:
                for lbar in Lbar:
                    if combinedVote[l] == 0:
                        lossCount +=1
                    elif combinedVote[l] <= combinedVote[lbar]:
                        lossCount += 1
            if len(Lbar)==0:
                lossCount = len(L)
            elif len(L)==0:
                lossCount = 0
            else:
                lossCount = lossCount/(len(L)*len(Lbar))
            return lossCount

        def multiLablePerformace(self, phenotypePrediction, true_phenotype, combinedVote):
            self.labelCount(phenotypePrediction, true_phenotype)
            self.intersection(phenotypePrediction, true_phenotype)
            self.union(phenotypePrediction, true_phenotype)
            self.loss_single = self.hammingLoss(phenotypePrediction, true_phenotype)
            self.precision_single = self.precision()
            self.recall_single = self.recall()
            self.accuracy_single = self.accuracy()
            self.f_measure_single = self.f_measure()
            if not combinedVote == None:
                self.oneError_single = self.oneError(combinedVote, true_phenotype)
                self.rankLoss_single = self.rankLoss(combinedVote, true_phenotype)

        def getLossSingle(self):
            return self.loss_single

        def getPrecisionSingle(self):
            return self.precision_single

        def getRecallSingle(self):
            return self.recall_single

        def getAccuracySingle(self):
            return self.accuracy_single

        def getFmeasureSingle(self):
            return self.f_measure_single

        def getOneErrorSingle(self):
            return self.oneError_single

        def getRankLossSingle(self):
            return self.rankLoss_single

        def getMacroF(self):
            return self.macro_f

        def getMicroF(self):
            return self.micro_f


        def reportMLperformance(self, measureList):
            print("Multi-label performance metrics report.......")
            print("-----------------------------------------------")
            for key, value in measureList.items():
                print(key + " measure value is: " + str(value) + "\n")


    class eLCS:
        def __init__(self):
            """ Initializes the eLCS algorithm """
            print("MLRBC: Initializing Algorithm...")
            # Global Parameters-------------------------------------------------------------------------------------
            self.population = None  # The rule population (the 'solution/model' evolved by eLCS)
            self.learnTrackOut = None  # Output file that will store tracking information during learning
            # -------------------------------------------------------
            # POPULATION REBOOT - Begin eLCS learning from an existing saved rule population
            # -------------------------------------------------------
            if cons.doPopulationReboot:
                self.correct = [0.0 for i in range(cons.trackingFrequency)]
                self.population = arg[-1]  # Population reboot
                self.doPopAnalysis()

                # self.population.runPopAveEval(self.exploreIter)
                # self.population.runAttGeneralitySum(True)
                cons.env.startEvaluationMode()  # Preserves learning position in training data
                if cons.testFile != 'None':  # If a testing file is available.
                    if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                        trainEval = self.doPopEvaluation(True)
                        testEval = self.doPopEvaluation(False)
                    else:
                        trainEval = self.doContPopEvaluation(True)
                        testEval = self.doContPopEvaluation(False)
                else:  # Only a training file is available
                    if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                        trainEval = self.doPopEvaluation(True)
                        testEval = None
                    else:
                        trainEval = self.doContPopEvaluation(True)
                        testEval = None

                if cons.ruleCompactionMethod != None:
                    RC = RuleCompaction(self.population, trainEval[1], testEval[1])
                    self.population = RC.pop

                    if cons.testFile != 'None':  # If a testing file is available.
                        if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                            trainEval = self.doPopEvaluation(True)
                            testEval = self.doPopEvaluation(False)
                        else:
                            trainEval = self.doContPopEvaluation(True)
                            testEval = self.doContPopEvaluation(False)
                    else:  # Only a training file is available
                        if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                            trainEval = self.doPopEvaluation(True)
                            testEval = None
                        else:
                            trainEval = self.doContPopEvaluation(True)
                            testEval = None

                OutputFileManager().writePopStats(cons.outFileName, trainEval, testEval, NO_TRAIN_ITERATION,
                                                  self.population, self.correct)

            # -------------------------------------------------------
            # NORMAL eLCS - Run eLCS from scratch on given data
            # -------------------------------------------------------
            else:
                try:
                    save_path = 'Run_results_MLRBC'
                    file_name = cons.outFileName + '_LearnTrack_' + str(exp) + '.txt'
                    completeName = os.path.join(save_path, file_name)
                    self.learnTrackOut = open(completeName, 'w')
                except Exception as inst:
                    print(type(inst))
                    print(inst.args)
                    print(inst)
                    print('cannot open', cons.outFileName + '_LearnTrack.txt')
                    raise
                else:
                    self.learnTrackOut.write("Explore_Iteration\tMacroPopSize\tMicroPopSize\tHamming_Loss\tAccuracy\tAveGenerality\tTime(min)\n")
                # Instantiate Population---------
                self.population = ClassifierSet()
                self.exploreIter = 0
                self.correct = [0.0 for i in range(cons.trackingFrequency)]
                self.hloss = [1 for i in range(cons.trackingFrequency)]
                # self.tp = [0.0 for i in range(cons.trackingFrequency)]
                # self.tn = [0.0 for i in range(cons.trackingFrequency)]
                # self.overGenAcc = [0.0 for i in range(cons.trackingFrequency)]

                # Run the eLCS algorithm-------------------------------------------------------------------------------
                self.run_eLCS()

        def run_eLCS(self):
            """ Runs the initialized eLCS algorithm. """
            # --------------------------------------------------------------
            print("Learning Checkpoints: " + str(cons.learningCheckpoints))
            print("Maximum Iterations: " + str(cons.maxLearningIterations))
            print("-----------------------------------------------------------------------------------------------------")
            loss_previous = 1.0

            # -------------------------------------------------------
            # MAJOR LEARNING LOOP
            # -------------------------------------------------------
            while self.exploreIter < int(cons.learningIterations)+1:
                if (self.exploreIter % cons.trackingFrequency) == 0:
                    print('Iteration: ', self.exploreIter)

                # -------------------------------------------------------
                # GET NEW INSTANCE AND RUN A LEARNING ITERATION
                # -------------------------------------------------------
                state_phenotype_conf = cons.env.getTrainInstance()
                self.runIteration(state_phenotype_conf, self.exploreIter)

                # -------------------------------------------------------------------------------------------------------------------------------
                # EVALUATIONS OF ALGORITHM
                # -------------------------------------------------------------------------------------------------------------------------------
                cons.timer.startTimeEvaluation()

                # -------------------------------------------------------
                # TRACK LEARNING ESTIMATES
                # -------------------------------------------------------

                if (self.exploreIter % cons.trackingFrequency) == (cons.trackingFrequency - 1) and self.exploreIter > 0:
                    self.population.runPopAveEval(self.exploreIter)
                    self.population.runAttGeneralitySum(True)
                    trackedAccuracy = sum(self.correct) / float(cons.trackingFrequency)  # Accuracy over the last "trackingFrequency" number of iterations.
                    trackedHloss = sum(self.hloss) / float(cons.trackingFrequency)
                    cons.env.startEvaluationMode()
                    testEval = self.doPopEvaluation(False)
                    d = abs(testEval[1]['HammingLoss'] - loss_previous)
                    if d < cons.error:
                        cons.learningIterations = self.exploreIter
                    else:
                        loss_previous = testEval[1]['HammingLoss']

                    # trackedTP = sum(self.tp) / float(cons.trackingFrequency)
                    # trackedTN = sum(self.tn) / float(cons.trackingFrequency)
                    # trackedOverGenAcc = sum(self.overGenAcc) / float(cons.trackingFrequency)
                    # self.learnTrackOut.write(self.population.getPopTrack(round(trackedHloss, 4), round(trackedAccuracy, 4), self.exploreIter + 1,
                    #                                                      cons.trackingFrequency, trackedTP, trackedTN, trackedOverGenAcc))
                    self.learnTrackOut.write(self.population.getPopTrack(round(trackedHloss, 4), round(trackedAccuracy, 4), self.exploreIter + 1))
                cons.timer.stopTimeEvaluation()

                # -------------------------------------------------------
                # CHECKPOINT - COMPLETE EVALUTATION OF POPULATION - strategy different for discrete vs continuous phenotypes
                # -------------------------------------------------------
                if (self.exploreIter) in [int(cons.learningIterations)]:
                    cons.timer.startTimeEvaluation()
                    print("-------------------------------------------------------------------------------------------------------------------")
                    print("Running Population Evaluation after " + str(self.exploreIter + 1) + " iterations.")

                    self.population.runPopAveEval(self.exploreIter)
                    self.population.runAttGeneralitySum(True)
                    cons.env.startEvaluationMode()  # Preserves learning position in training data
                    if cons.testFile != 'None':  # If a testing file is available.
                        if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                            trainEval = self.doPopEvaluation(True)
                            testEval = self.doPopEvaluation(False)
                        else:
                            trainEval = self.doContPopEvaluation(True)
                            testEval = self.doContPopEvaluation(False)
                    else:  # Only a training file is available
                        if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                            trainEval = self.doPopEvaluation(True)
                            testEval = None
                        else:
                            trainEval = self.doContPopEvaluation(True)
                            testEval = None

                    cons.env.stopEvaluationMode()  # Returns to learning position in training data
                    cons.timer.stopTimeEvaluation()
                    cons.timer.returnGlobalTimer()

                    if cons.ruleCompactionMethod != None:
                        RC = RuleCompaction(self.population, trainEval[1], testEval[1])
                        self.population = RC.pop

                        print(
                            "----------------------------------------------------------------")
                        print("Running Population Evaluation after rule compaction...")

                        self.population.runPopAveEval(self.exploreIter)
                        self.population.runAttGeneralitySum(True)
                        cons.env.startEvaluationMode()  # Preserves learning position in training data
                        if cons.testFile != 'None':  # If a testing file is available.
                            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                                trainEval = self.doPopEvaluation(True)
                                testEval = self.doPopEvaluation(False)
                            else:
                                trainEval = self.doContPopEvaluation(True)
                                testEval = self.doContPopEvaluation(False)
                        else:  # Only a training file is available
                            if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                                trainEval = self.doPopEvaluation(True)
                                testEval = None
                            else:
                                trainEval = self.doContPopEvaluation(True)
                                testEval = None

                    # Write output files----------------------------------------------------------------------------------------------------------
                    OutputFileManager().writePopStats(cons.outFileName, trainEval, testEval, self.exploreIter + 1,
                                                      self.population, self.correct)
                    OutputFileManager().writePop(cons.outFileName, self.exploreIter + 1, self.population)
                    # ----------------------------------------------------------------------------------------------------------------------------

                # -------------------------------------------------------
                # ADJUST MAJOR VALUES FOR NEXT ITERATION
                # -------------------------------------------------------
                self.exploreIter += 1  # Increment current learning iteration
                cons.env.newInstance(True)  # Step to next instance in training set

            # Once eLCS has reached the last learning iteration, close the tracking file
            self.learnTrackOut.close()
            print("MLRBC Run Completed on model " + str(arg[0]))

        def runIteration(self, state_phenotype_conf, exploreIter):
            # -----------------------------------------------------------------------------------------------------------------------------------------
            # FORM A MATCH SET - includes covering
            # -----------------------------------------------------------------------------------------------------------------------------------------
            self.population.makeMatchSet(state_phenotype_conf, exploreIter)
            # -----------------------------------------------------------------------------------------------------------------------------------------
            # MAKE A PREDICTION - utilized here for tracking estimated learning progress.  Typically used in the explore phase of many LCS algorithms.
            # -----------------------------------------------------------------------------------------------------------------------------------------
            Acc = ClassAccuracy()
            cons.timer.startTimeEvaluation()
            prediction = Prediction(self.population)
            prediction.calMaxPrediction()
            phenotypePrediction = prediction.getDecision()

            itt = exploreIter % cons.trackingFrequency
            # -------------------------------------------------------
            # PREDICTION NOT POSSIBLE
            # -------------------------------------------------------
            # self.overGenAcc[itt] = self.population.overGenAcc
            if phenotypePrediction == None or phenotypePrediction == 'Tie':
                if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                    phenotypePrediction = random.choice(cons.env.formatData.phenotypeList)
                else:
                    phenotypePrediction = random.randrange(cons.env.formatData.phenotypeList[0],
                                                           cons.env.formatData.phenotypeList[1], (
                                                           cons.env.formatData.phenotypeList[1] -
                                                           cons.env.formatData.phenotypeList[0]) / float(1000))
            else:  # Prediction Successful
                # -------------------------------------------------------
                # DISCRETE PHENOTYPE PREDICTION
                # -------------------------------------------------------
                if cons.env.formatData.discretePhenotype or cons.env.formatData.MLphenotype:
                    target = state_phenotype_conf[1]
                    tp = 0
                    tn = 0
                    """
                    for it in range(cons.env.formatData.ClassCount):
                        tp += int(target[it]) and int(phenotypePrediction[it])
                        tn += not (int(target[it]) or int(phenotypePrediction[it]))
                    self.tp[itt] = tp
                    self.tn[itt] = tn
                    """
                    target_int = int(state_phenotype_conf[1], 2)
                    phenotypePrediction_int = int(phenotypePrediction, 2)
                    if phenotypePrediction_int == target_int:
                        self.correct[itt] = 1
                        # self.hloss[itt] = 0
                    else:
                        self.correct[itt] = 0
                        self.hloss[itt] = Acc.hammingLoss(phenotypePrediction, state_phenotype_conf[1])
                # -------------------------------------------------------
                # CONTINUOUS PHENOTYPE PREDICTION
                # -------------------------------------------------------
                else:
                    predictionError = math.fabs(phenotypePrediction - float(state_phenotype_conf[1]))
                    phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
                    accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
                    self.correct[itt] = accuracyEstimate

            cons.timer.stopTimeEvaluation()
            # -----------------------------------------------------------------------------------------------------------------------------------------
            # FORM A CORRECT SET
            # -----------------------------------------------------------------------------------------------------------------------------------------
            self.population.makeCorrectSet(state_phenotype_conf[1])
            # -----------------------------------------------------------------------------------------------------------------------------------------
            # UPDATE PARAMETERS
            # -----------------------------------------------------------------------------------------------------------------------------------------
            self.population.updateSets(exploreIter, state_phenotype_conf)
            # -----------------------------------------------------------------------------------------------------------------------------------------
            # SUBSUMPTION - APPLIED TO CORRECT SET - A heuristic for addition additional generalization pressure to eLCS
            # -----------------------------------------------------------------------------------------------------------------------------------------
            if cons.doSubsumption:
                cons.timer.startTimeSubsumption()
                self.population.doCorrectSetSubsumption()
                cons.timer.stopTimeSubsumption()
            # -----------------------------------------------------------------------------------------------------------------------------------------
            # RUN THE GENETIC ALGORITHM - Discover new offspring rules from a selected pair of parents
            # -----------------------------------------------------------------------------------------------------------------------------------------
            if len(self.population.correctSet)>=1:
                self.population.runGA(exploreIter, state_phenotype_conf[0], state_phenotype_conf[1])
            # -----------------------------------------------------------------------------------------------------------------------------------------
            # SELECT RULES FOR DELETION - This is done whenever there are more rules in the population than 'N', the maximum population size.
            # -----------------------------------------------------------------------------------------------------------------------------------------
            self.population.deletion(exploreIter)
            self.population.clearSets()  # Clears the match and correct sets for the next learning iteration

        def doPopEvaluation(self, isTrain):

            if isTrain:
                myType = "TRAINING"
            else:
                myType = "TESTING"
            noMatch = 0
            tie = 0
            Hloss = 0
            precision = 0
            accuracy = 0
            recall = 0
            fmeasure = 0
            oneError = 0
            rankLoss = 0
            MLperformance = {}


            cons.env.resetDataRef(isTrain)  # Go to the first instance in dataset
            Acc = ClassAccuracy()

            if isTrain:
                instances = cons.env.formatData.numTrainInstances
            else:
                instances = cons.env.formatData.numTestInstances
            labelList = np.empty([instances, cons.env.formatData.ClassCount])
            targetList = np.empty([instances, cons.env.formatData.ClassCount])
            noPrediction = np.zeros(cons.env.formatData.ClassCount)

            if THRESHOLD == 'pcut' and PREDICTION_METHOD == 'agg':
                for inst in range(instances):
                    if isTrain:
                        state_phenotype_conf = cons.env.getTrainInstance()
                    else:
                        state_phenotype_conf = cons.env.getTestInstance()
                    self.population.makeEvalMatchSet(state_phenotype_conf[0])
                    prediction = Prediction(self.population)
                    prediction.combinePredictions()
                    combVote = prediction.getCombVote()
                    labelList[inst] = np.array(combVote)
                    cons.env.newInstance(isTrain)
                    self.population.clearSets()
                    targetList[inst] = [int(l) for l in list(state_phenotype_conf[1])]

                prediction.pcut(np.array(labelList), instances)
                combPredList = prediction.getCombPred()

                for inst in range(instances):
                    if isTrain:
                        state_phenotype_conf = cons.env.getTrainInstance()
                    else:
                        state_phenotype_conf = cons.env.getTestInstance()
                    combPred = combPredList[inst]
                    if combPred == None:
                        labelList[inst] = noPrediction
                        noMatch += 1
                    elif combPred == 'Tie':
                        labelList[inst] = noPrediction
                        tie += 1
                    else:
                        Acc.multiLablePerformace(combPred, state_phenotype_conf[1], labelList[inst].tolist())
                        """
                        Sample-based metrics
                        """
                        Hloss += Acc.getLossSingle()
                        precision += Acc.getPrecisionSingle()
                        accuracy += Acc.getAccuracySingle()
                        recall += Acc.getRecallSingle()
                        fmeasure += Acc.getFmeasureSingle()
                        oneError += Acc.getOneErrorSingle()
                        rankLoss += Acc.getRankLossSingle()
                        """
                        Class-based metrics
                        """
                        Acc.updateClassBased(state_phenotype_conf[1], combPred)
                    cons.env.newInstance(isTrain)
            else:
                for inst in range(instances):
                    if isTrain:
                        state_phenotype_conf = cons.env.getTrainInstance()
                    else:
                        state_phenotype_conf = cons.env.getTestInstance()
                    self.population.makeEvalMatchSet(state_phenotype_conf[0])
                    prediction = Prediction(self.population)

                    if PREDICTION_METHOD == 'max':
                        prediction.calMaxPrediction()
                        combPred = prediction.getDecision()
                        combVote = [0] * cons.env.formatData.ClassCount
                    elif THRESHOLD == 'onethreshold':
                        prediction.combinePredictions()
                        prediction.oneThreshold()
                        combPred = prediction.getCombPred()
                        combVote = prediction.getCombVote()
                    elif THRESHOLD == 'rcut':
                        prediction.combinePredictions()
                        prediction.rcut()
                        combPred = prediction.getCombPred()
                        combVote = prediction.getCombVote()

                    targetList[inst] = [int(l) for l in list(state_phenotype_conf[1])]  # The list of all target labels of the dataset

                    if combPred == None:
                        labelList[inst] = noPrediction
                        noMatch += 1
                    elif combPred == 'Tie':
                        labelList[inst] = noPrediction
                        tie += 1
                    else:
                        Acc.multiLablePerformace(combPred, state_phenotype_conf[1], combVote)
                        labelList[inst] = np.array(combVote)
                        """
                        Sample-based metrics
                        """
                        Hloss += Acc.getLossSingle()
                        precision += Acc.getPrecisionSingle()
                        accuracy += Acc.getAccuracySingle()
                        recall += Acc.getRecallSingle()
                        fmeasure += Acc.getFmeasureSingle()
                        oneError += Acc.getOneErrorSingle()
                        rankLoss += Acc.getRankLossSingle()
                        """
                        Class-based metrics
                        """
                        Acc.updateClassBased(state_phenotype_conf[1], combPred)

                    cons.env.newInstance(isTrain)
                    self.population.clearSets()

            """
            micro-average metrics
            """
            Acc.microAverage()

            """
            macro-average metrics
            """
            fpr = dict()
            tpr = dict()
            roc_auc = np.zeros(cons.env.formatData.ClassCount)
            Acc.macroAverage()
            for l in range(cons.env.formatData.ClassCount):
                fpr[l], tpr[l], _ = roc_curve(targetList[:, l], labelList[:, l])
                roc_auc[l] = auc(fpr[l], tpr[l])
            where_are_NaNs = np.isnan(roc_auc)
            roc_auc[where_are_NaNs] = 0.0

            MLperformance["HammingLoss"] = Hloss/float(instances)
            MLperformance["Precision"] = precision/float(instances)
            MLperformance["Accuracy"] = accuracy/float(instances)
            MLperformance["Recall"] = recall/float(instances)
            MLperformance["F-measure"] = fmeasure/float(instances)
            MLperformance["oneError"] = oneError/float(instances)
            MLperformance["rankLoss"] = rankLoss/float(instances)
            MLperformance["micro-F"] = Acc.getMicroF()
            MLperformance["macro-F"] = Acc.getMacroF()
            MLperformance["AUC-score"] = roc_auc.mean()
            self.perfreport = MLperformance

            predictionFail = float(noMatch) / float(instances)
            predictionTies = float(tie) / float(instances)
            instanceCoverage = 1.0 - predictionFail
            predictionMade = 1.0 - (predictionFail + predictionTies)

            # print("-----------------------------------------------")
            # print(str(myType) + " Evaluation Results on model " + str(arg[0]) +  ":")
            resultList = [instanceCoverage, MLperformance]
            # print(resultList[0], resultList[1])

            return resultList

        def doContPopEvaluation(self, isTrain):
            """ Performs evaluation of population via the copied environment. Specifically developed for continuous phenotype evaulation.
            The population is maintained unchanging throughout the evaluation.  Works on both training and testing data. """
            if isTrain:
                myType = "TRAINING"
            else:
                myType = "TESTING"
            noMatch = 0  # How often does the population fail to have a classifier that matches an instance in the data.
            cons.env.resetDataRef(isTrain)  # Go to first instance in data set
            accuracyEstimateSum = 0

            if isTrain:
                instances = cons.env.formatData.numTrainInstances
            else:
                instances = cons.env.formatData.numTestInstances
            # ----------------------------------------------------------------------------------------------
            for inst in range(instances):
                if isTrain:
                    state_phenotype = cons.env.getTrainInstance()
                else:
                    state_phenotype = cons.env.getTestInstance()
                # -----------------------------------------------------------------------------
                self.population.makeEvalMatchSet(state_phenotype[0])
                prediction = Prediction(self.population)
                phenotypePrediction = prediction.getDecision()
                # -----------------------------------------------------------------------------
                if phenotypePrediction == None:
                    noMatch += 1
                else:  # Instances which failed to be covered are excluded from the initial accuracy calculation
                    predictionError = math.fabs(float(phenotypePrediction) - float(state_phenotype[1]))
                    phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
                    accuracyEstimateSum += 1.0 - (predictionError / float(phenotypeRange))

                cons.env.newInstance(isTrain)  # next instance
                self.population.clearSets()
                # ----------------------------------------------------------------------------------------------
            # Accuracy Estimate
            if instances == noMatch:
                accuracyEstimate = 0
            else:
                accuracyEstimate = accuracyEstimateSum / float(instances - noMatch)

            # Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
            instanceCoverage = 1.0 - (float(noMatch) / float(instances))
            adjustedAccuracyEstimate = accuracyEstimateSum / float(
                instances)  # noMatchs are treated as incorrect predictions (can see no other fair way to do this)

            print("-----------------------------------------------")
            print(str(myType) + " Evaluation Results:-------------")
            print("Instance Coverage = " + str(instanceCoverage * 100.0) + '%')
            print("Estimated Prediction Accuracy (Ignore uncovered) = " + str(accuracyEstimate))
            print("Estimated Prediction Accuracy (Penalty uncovered) = " + str(adjustedAccuracyEstimate))
            # Balanced and Standard Accuracies will only be the same when there are equal instances representative of each phenotype AND there is 100% covering.
            resultList = [adjustedAccuracyEstimate, instanceCoverage]
            return resultList

        def doPopAnalysis(self):
            initialCount = len(self.population.popSet)
            newPopSet = []
            for cl in self.population.popSet:
                if cl.matchCount > 1:
                    newPopSet.append(cl)
            newCount = len(newPopSet)
            self.population.popSet = newPopSet

            # dict = {}
            # for cl in self.population.popSet:
            #     if cl.phenotype in dict.keys():
            #         acc = dict[cl.phenotype]
            #         acc.append(cl.accuracy)
            #         dict[cl.phenotype] = acc
            #     else:
            #         acc = []
            #         acc.append(cl.accuracy)
            #         dict[cl.phenotype] = acc
            # info = {}
            # for label in dict.keys():
            #     maxx = max(dict[label])
            #     meann = np.mean(dict[label])
            #     minn = min(dict[label])
            #     info[label] = [maxx, meann, minn]



    class OutputFileManager:

        def writePopStats(self, outFile, trainEval, testEval, exploreIter, pop, correct):
            """ Makes output text file which includes all of the evaluation statistics for a complete analysis of all training and testing data on the current eLCS rule population. """
            try:
                save_path = 'Run_results_MLRBC'
                completeName = os.path.join(save_path, outFile)
                popStatsOut = open(completeName + '_' + str(exploreIter) + '_PopStats_' + str(exp) + '.txt', 'w')
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', outFile + '_' + str(exploreIter) + '_PopStats_' + str(exp) + '.txt')
                raise

            # Evaluation of pop
            popStatsOut.write(
                "Performance Statistics:------------------------------------------------------------------------\n")
            popStatsOut.write("Training Coverage\tTesting Coverage\tTraining Performance\tTest Performance\n")

            if cons.testFile != 'None':
                popStatsOut.write(str(trainEval[0]) + "\t")
                popStatsOut.write(str(testEval[0]) + "\t")
                popStatsOut.write(str(trainEval[1]) + "\t")
                popStatsOut.write(str(testEval[1]) + "\n\n")
            elif cons.trainFile != 'None':
                popStatsOut.write(str(trainEval[0]) + "\t")
                popStatsOut.write("NA\t")
                popStatsOut.write(str(trainEval[1]) + "\t")
                popStatsOut.write("NA\n\n")
            else:
                popStatsOut.write("NA\t")
                popStatsOut.write("NA\t")
                popStatsOut.write("NA\t")
                popStatsOut.write("NA\t")
                popStatsOut.write("NA\t")
                popStatsOut.write("NA\n\n")

            popStatsOut.write(
                "Population Characterization:------------------------------------------------------------------------\n")
            popStatsOut.write("MacroPopSize\tMicroPopSize\tGenerality\n")
            popStatsOut.write(str(len(pop.popSet)) + "\t" + str(pop.microPopSize) + "\t" + str("%.2f" % pop.aveGenerality) + "\n\n")

            popStatsOut.write("SpecificitySum:------------------------------------------------------------------------\n")
            headList = cons.env.formatData.trainHeaderList  # preserve order of original dataset

            for i in range(len(headList)):
                if i < len(headList) - 1:
                    popStatsOut.write(str(headList[i]) + "\t")
                else:
                    popStatsOut.write(str(headList[i]) + "\n")

            # Prints out the Specification Sum for each attribute
            for i in range(len(pop.attributeSpecList)):
                if i < len(pop.attributeSpecList) - 1:
                    popStatsOut.write(str(pop.attributeSpecList[i]) + "\t")
                else:
                    popStatsOut.write(str(pop.attributeSpecList[i]) + "\n")

            popStatsOut.write("\nAccuracySum:------------------------------------------------------------------------\n")
            for i in range(len(headList)):
                if i < len(headList) - 1:
                    popStatsOut.write(str(headList[i]) + "\t")
                else:
                    popStatsOut.write(str(headList[i]) + "\n")

            # Prints out the Accuracy Weighted Specification Count for each attribute
            for i in range(len(pop.attributeAccList)):
                if i < len(pop.attributeAccList) - 1:
                    popStatsOut.write(str(pop.attributeAccList[i]) + "\t")
                else:
                    popStatsOut.write(str(pop.attributeAccList[i]) + "\n")

                    # Time Track ---------------------------------------------------------------------------------------------------------
            popStatsOut.write(
                "\nRun Time(in minutes):------------------------------------------------------------------------\n")
            popStatsOut.write(cons.timer.reportTimes())
            popStatsOut.close()

        def writePop(self, outFile, exploreIter, pop):
            """ Writes a tab delimited text file outputting the entire evolved rule population, including conditions, phenotypes, and all rule parameters. """
            try:
                save_path = 'Run_results_MLRBC'
                completeName = os.path.join(save_path, outFile)
                rulePopOut = open(completeName + '_RulePop_' + str(exp) + '.txt', 'w')
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', outFile + '_RulePop_' + str(exp) + '.txt')
                raise

            # Write Header-----------------------------------------------------------------------------------------------------------------------------------------------
            dataLink = cons.env.formatData
            headList = dataLink.trainHeaderList
            for i in range(len(headList)):
                rulePopOut.write(str(headList[i]) + "\t")
            rulePopOut.write(
                "Phenotype\tFitness\tAccuracy\tNumerosity\tAveMatchSetSize\tTimeStampGA\tInitTimeStamp\tSpecificity\tDeletionProb\tCorrectCount\tMatchCount\n")

            # Write each classifier--------------------------------------------------------------------------------------------------------------------------------------
            for cl in pop.popSet:
                rulePopOut.write(str(cl.printClassifier()))

            rulePopOut.close()


    class RuleCompaction:
        def __init__(self, pop, originalTrainAcc, originalTestAcc):
            """ Initialize and run the specified rule compaction strategy. """
            print(
                "------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Starting Rule Compaction Algorithm (" + str(cons.ruleCompactionMethod) + ") on Model " + str(exp)+ "...")
            self.pop = pop
            self.originalTrainLoss = originalTrainAcc['HammingLoss']
            self.originalTestLoss = originalTestAcc['HammingLoss']

            # print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            # print("Original Training Accuracy = " + str(self.originalTrainLoss))
            # print("Original test Loss = " + str(self.originalTestLoss))

            # Rule Compaction Strategies Included in the Paper----------------------------------------------------------------------------------------------------------------
            # Strategy Naming Key: (Code Name = Paper Draft Name = Final Paper Name)

            # A8 = Dixion = CRA2
            # A9 = UCRA = PDRC
            # A12 = Fu1 = Fu1
            # A13 = Fu2 = Fu2
            # A17 = QCRA = PDRC
            # QRC = QRC = QRF

            if cons.ruleCompactionMethod == 'A8':  # Existing Published Strategy: Based on Dixon and Shoeleh's method, use the product of accuracy and generality
                self.Approach8()
            elif cons.ruleCompactionMethod == 'A9':  # Use the product of accuracy, generality and numerosity
                self.Approach9()
            elif cons.ruleCompactionMethod == 'A12':  # Existing Published Strategy: Completely follow Fu's approach 1
                self.Approach12()
            elif cons.ruleCompactionMethod == 'A13':  # Existing Published Strategy: Completely follow Fu's approach 2
                self.Approach13()
            elif cons.ruleCompactionMethod == 'A17':  # Quick fitness covering
                self.Approach17()
            elif cons.ruleCompactionMethod == 'QRC':
                self.QuickRuleCleanup()

            # Unused Early Experimental Rule Compaction Strategies:-------------------------------------------------------------------------------------------------------------
            elif cons.ruleCompactionMethod == 'A1':  # Training Accuracy Based (Stage 1,2,3), Stage 3 uses match covering to rank rules
                self.Approach1()
            elif cons.ruleCompactionMethod == 'A2':  # Training Accuracy Based (Stage 1,2,3), Stage 3 uses fitness to rank rules
                self.Approach2()
            elif cons.ruleCompactionMethod == 'A3':  # Training Accuracy Based (Stage 1,2), Light Compaction - NO STAGE 3
                self.Approach3()
            elif cons.ruleCompactionMethod == 'A4':  # Testing Accuracy Based (Stage 1,2), Light Compaction - NO STAGE 3
                self.Approach4()
            elif cons.ruleCompactionMethod == 'A5':  # Experimental - remove likely bad rules without evaluations then follow with standard Training Accuracy Based State 1, and 2.
                self.Approach5()
            elif cons.ruleCompactionMethod == 'A6':  # Based on Dixon's method, only use accuracy
                self.Approach6()
            elif cons.ruleCompactionMethod == 'A7':  # Based on Dixon's method, use the product of accuracy and numerosity
                self.Approach7()
            elif cons.ruleCompactionMethod == 'A10':  # Rule's entropy method introduced by Kharbat
                self.Approach10()
            elif cons.ruleCompactionMethod == 'A11':  # stage 1: QRC, stage 2: match covering (same to A1)
                self.Approach11()
            elif cons.ruleCompactionMethod == 'A14':  # stage 1: QRC, stage 2: use fitness to rank rules and do covering
                self.Approach14()
            elif cons.ruleCompactionMethod == 'A15':  # Only use fitness to rank rules and do covering
                self.Approach15()
            elif cons.ruleCompactionMethod == 'A16':  # Only use numerosity to rank rules and do covering
                self.Approach16()
            elif cons.ruleCompactionMethod == 'A18':  # Stage 1,2 + Quick fitness covering
                self.Approach18()
            else:
                print("RuleCompaction: Error - specified rule compaction strategy not found.")

        ############################################################################################################################################################################################
        def Approach8(self):
            """ This approach is based on Dixon's method. For each sample, form a match set and then a correct set. The most useful rule in
                the correct set is moved into the final ruleset. In this approach, the most useful rule has the largest product of accuracy
                and generality."""

            retainedClassifiers = []
            self.matchSet = []
            self.correctSet = []

            cons.env.startEvaluationMode()
            cons.env.resetDataRef(True)
            for j in range(cons.env.formatData.numTrainInstances):
                state_class = cons.env.getTrainInstance()
                state = state_class[0]
                classification = state_class[1]

                # Create MatchSet
                for i in range(len(self.pop.popSet)):
                    cl = self.pop.popSet[i]
                    if cl.match(state):
                        self.matchSet.append(i)

                # Create CorrectSet
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if self.pop.popSet[ref].phenotype == classification:
                        self.correctSet.append(ref)

                # Find the rule with highest accuracy, generality product
                highestValue = 0
                highestRef = 0
                for i in range(len(self.correctSet)):
                    ref = self.correctSet[i]
                    product = self.pop.popSet[ref].accuracy * (
                    cons.env.formatData.numAttributes - len(self.pop.popSet[ref].condition)) / float(
                        cons.env.formatData.numAttributes)
                    if product > highestValue:
                        highestValue = product
                        highestRef = ref

                # If the rule is not already in the final ruleset, move it to the final ruleset
                if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[highestRef])

                # Move to the next instance
                cons.env.newInstance(True)
                self.matchSet = []
                self.correctSet = []
            cons.env.stopEvaluationMode()

            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))
            return self.pop

        ############################################################################################################################################################################################
        def Approach9(self):
            """ This approach is based on Dixon's approach, called UCRA in the paper. For each sample, form a match set and then a correct set.
            The most useful rule in the correct set is moved into the final ruleset. In this approach, the most useful rule has the largest
            product of accuracy, numerosity and generality."""

            retainedClassifiers = []
            self.matchSet = []
            self.correctSet = []

            cons.env.startEvaluationMode()
            cons.env.resetDataRef(True)
            for j in range(cons.env.formatData.numTrainInstances):
                state_class = cons.env.getTrainInstance()
                state = state_class[0]
                classification = state_class[1]

                # Create MatchSet
                for i in range(len(self.pop.popSet)):
                    cl = self.pop.popSet[i]
                    if cl.match(state):
                        self.matchSet.append(i)

                # Create CorrectSet
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if self.pop.popSet[ref].phenotype == classification:
                        self.correctSet.append(ref)

                # Find the rule with highest accuracy, generality and numerosity product
                highestValue = 0.0
                highestRef = 0.0
                for i in range(len(self.correctSet)):
                    ref = self.correctSet[i]
                    product = self.pop.popSet[ref].accuracy * (
                    cons.env.formatData.numAttributes - len(self.pop.popSet[ref].condition)) / float(
                        cons.env.formatData.numAttributes) * self.pop.popSet[ref].numerosity
                    if product > highestValue:
                        highestValue = product
                        highestRef = ref

                # If the rule is not already in the final ruleset, move it to the final ruleset
                if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[highestRef])

                # Move to the next instance
                cons.env.newInstance(True)
                self.matchSet = []
                self.correctSet = []
            cons.env.stopEvaluationMode()

            self.pop.popSet = retainedClassifiers
            count = 0
            for cl in self.pop.popSet:
                count += cl.numerosity
            self.pop.microPopSize = count

            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach12(self):
            """ This approach completely follows Fu's first approach. In the third stage, the number of samples a rule matched is used to rank
        	the rules and guide covering. Ranking list is updated each time some samples are covered and removed from the training set. """
            # Order Classifier Set---------------------------------------------------------------------------------------------------------

            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            #        for i in range(len(self.pop.popSet)):
            #            print self.pop.popSet[i].numerosity
            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                #print(newAccuracy)
                if newAccuracy < self.originalTrainAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc
            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                #print(newAccuracy)
                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 3----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            completelyGeneralRuleRef = None
            if len(self.pop.popSet) == 0:  # Stop Check
                keepGoing = False
            else:
                keepGoing = True

            # Make the match count list in preparation for state 3-------------------------------------------------------------------------
            matchCountList = [0.0 for v in range(len(self.pop.popSet))]
            cons.env.startEvaluationMode()
            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                cons.env.resetDataRef(True)
                for j in range(cons.env.formatData.numTrainInstances):  # For each instance in training data
                    cl = self.pop.popSet[i]
                    state = cons.env.getTrainInstance()[0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchCountList[i] += 1
                    cons.env.newInstance(True)
                if len(self.pop.popSet[i].condition) == 0:
                    completelyGeneralRuleRef = i
            #print(matchCountList)
            cons.env.stopEvaluationMode()
            if completelyGeneralRuleRef != None:  # gets rid of completely general rule.
                del matchCountList[completelyGeneralRuleRef]
                del self.pop.popSet[completelyGeneralRuleRef]

            # ----------------------------------------------------------------------------------------------------------------------------
            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted
            while len(trainingData) > 0 and keepGoing:
                bestRef = None
                bestValue = None
                for i in range(len(matchCountList)):
                    if bestValue == None or bestValue < matchCountList[i]:
                        bestRef = i
                        bestValue = matchCountList[i]

                if bestValue == 0.0 or len(self.pop.popSet) < 1:
                    keepGoing = False
                    continue
                print(bestRef, bestValue)

                # Update Training Data----------------------------------------------------------------------------------------------------
                matchedData = 0
                w = 0
                cl = self.pop.popSet[bestRef]
                for i in range(len(trainingData)):
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                        del trainingData[w]
                    else:
                        w += 1
                if matchedData > 0:
                    # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                    finalClassifiers.append(self.pop.popSet[bestRef])

                # Update classifier list
                del self.pop.popSet[bestRef]

                # re-calculate match count list
                matchCountList = [0.0 for v in range(len(self.pop.popSet))]
                for i in range(len(self.pop.popSet)):
                    dataRef = 0
                    for j in range(len(trainingData)):  # For each instance in training data
                        cl = self.pop.popSet[i]
                        state = trainingData[dataRef][0]
                        doesMatch = cl.match(state)
                        if doesMatch:
                            matchCountList[i] += 1
                        dataRef += 1
                print(matchCountList)

                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 3 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach13(self):
            """ This approach completely follows Fu's second approach. All three stages use accuracy to sort rules."""

            # Order Classifier Set---------------------------------------------------------------------------------------------------------
            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)  # DEBUG
                if newAccuracy < self.originalTrainAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc
            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation

                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 3----------------------------------------------------------------------------------------------------------------------
            Sort_popSet = sorted(self.pop.popSet, key=self.numerositySort, reverse=True)
            self.pop.popSet = Sort_popSet[:]
            RefAccuary = self.performanceEvaluation(True)

            if len(self.pop.popSet) == 0:  # Stop check
                keepGoing = False
            else:
                keepGoing = True

            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation

                if newAccuracy < RefAccuracy:
                    self.pop.popSet.append(heldClassifier)
                else:
                    RefAccuracy = newAccuracy

            print("STAGE 3 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def QuickRuleCleanup(self):
            """ An extremely fast rule compaction strategy. Removes any rule with an accuracy below 50% and any rule that covers only one instance, but specifies more than one attribute
             (won't get rid of rare variant rules)"""

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            for i in range(len(self.pop.popSet)):
                if self.pop.popSet[i].accuracy <= 0.5 or (
                        self.pop.popSet[i].correctCount == 1 and len(self.pop.popSet[i].specifiedAttList) > 1):
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[i])
            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))
            return self.pop

        ############################################################################################################################################################################################
        def Approach17(self):
            """Called QCRA in the paper. It uses fitness to rank rules and guide covering. It's the same as Approach 15, but the code is re-written in
            order to speed up."""

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            print("Original Testing Accuracy = " + str(self.originalTestAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            if len(self.pop.popSet) == 0:  # Stop check
                keepGoing = False
            else:
                keepGoing = True

            lastGood_popSet = sorted(self.pop.popSet, key=self.accuracySort, reverse=True)
            self.pop.popSet = lastGood_popSet[:]

            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted

            while len(trainingData) > 0 and keepGoing:
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[0]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                0])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[0]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach1(self):
            """ A CRA based approach introduced by Fu 2000. The first two stages are the same as Fu's approach. The third stage first use how
            many samples a rule matched to rank the rule. Then the covering is done based on the ranking. The ranking list is not updated each
            time some samples are covered and removed from the training set. """
            # Order Classifier Set---------------------------------------------------------------------------------------------------------
            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)  # DEBUG
                if newAccuracy < self.originalTrainAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc
            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)
                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 3----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            completelyGeneralRuleRef = None
            if len(self.pop.popSet) == 0:  # Stop Check
                keepGoing = False
            else:
                keepGoing = True

            # Make the match count list in preparation for state 3-------------------------------------------------------------------------
            matchCountList = [0.0 for v in range(len(self.pop.popSet))]
            cons.env.startEvaluationMode()
            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                cons.env.resetDataRef(True)
                for j in range(cons.env.formatData.numTrainInstances):  # For each instance in training data
                    cl = self.pop.popSet[i]
                    state = cons.env.getTrainInstance()[0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchCountList[i] += 1
                    cons.env.newInstance(True)
                if len(self.pop.popSet[i].condition) == 0:
                    completelyGeneralRuleRef = i

            cons.env.stopEvaluationMode()
            if completelyGeneralRuleRef != None:  # gets rid of completely general rule.
                del matchCountList[completelyGeneralRuleRef]
                del self.pop.popSet[completelyGeneralRuleRef]

            # ----------------------------------------------------------------------------------------------------------------------------
            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted
            while len(trainingData) > 0 and keepGoing:
                bestRef = None
                bestValue = None
                for i in range(len(matchCountList)):
                    if bestValue == None or bestValue < matchCountList[i]:
                        bestRef = i
                        bestValue = matchCountList[i]

                if bestValue == 0.0 or len(self.pop.popSet) < 1:
                    keepGoing = False
                    continue

                # Update Training Data----------------------------------------------------------------------------------------------------
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[bestRef]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                bestRef])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[bestRef]
                del matchCountList[bestRef]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 3 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach2(self):
            """ A variant of CRA based approach introduced by Fu 2000. The first two stages are the same as Fu's approach. In the third stage,
            fitness is used to rank the rules and guide covering. Ranking list is updated each time some samples are covered and removed from
            the training set. """

            # Order Classifier Set---------------------------------------------------------------------------------------------------------
            print(self.pop.popSet)
            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            print(self.pop.popSet)
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)  # DEBUG
                if newAccuracy < self.originalTrainAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc
            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation

                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 3----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            completelyGeneralRuleRef = None
            if len(self.pop.popSet) == 0:  # Stop check
                keepGoing = False
            else:
                keepGoing = True

            fitnessList = [0.0 for v in range(len(self.pop.popSet))]
            for i in range(len(self.pop.popSet)):
                fitnessList[i] = self.pop.popSet[i].fitness

            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                if len(self.pop.popSet[i].condition) == 0:
                    completelyGeneralRuleRef = i

            if completelyGeneralRuleRef != None:  # gets rid of completely general rule.
                del fitnessList[completelyGeneralRuleRef]
                del self.pop.popSet[completelyGeneralRuleRef]
            # ----------------------------------------------------------------------------------------------------------------------------
            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted
            while len(trainingData) > 0 and keepGoing:
                bestRef = None
                bestValue = None
                for i in range(len(fitnessList)):
                    if bestValue == None or bestValue < fitnessList[i]:
                        bestRef = i
                        bestValue = fitnessList[i]

                if bestValue == 0.0 or len(self.pop.popSet) < 1:
                    keepGoing = False
                    continue

                # Update Training Data----------------------------------------------------------------------------------------------------
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[bestRef]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                bestRef])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[bestRef]
                del fitnessList[bestRef]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 3 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach3(self):
            """ A light Rule compaction algorithm - Uses Training accuracy throughout, ranks by numerosity originally
            leaves out stage 3 - intended to just remove major noisy rules """

            # Order Classifier Set---------------------------------------------------------------------------------------------------------
            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)  # DEBUG
                if newAccuracy < self.originalTrainAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc

            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach4(self):
            """ A light Rule compaction algorithm  Uses Testing accuracy throughout, ranks by numerosity originally
            leaves out stage 3 - intended to make a maximally small rule set. """

            # Order Classifier Set---------------------------------------------------------------------------------------------------------
            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTestAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(False)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)  # DEBUG
                if newAccuracy < self.originalTestAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc
            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(False)  # Perform classifier set test accuracy evaluation

                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach5(self):
            """ Experimental Approach: remove likely bad rules without evaluations then follow with standard Training Accuracy Based State 1,
            and 2. """

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            print("Original Testing Accuracy = " + str(self.originalTestAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            for i in range(len(self.pop.popSet)):
                if self.pop.popSet[i].getAccuracy() > 0.5:
                    retainedClassifiers.append(self.pop.popSet[i])
            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            # Order Classifier Set---------------------------------------------------------------------------------------------------------
            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                if newAccuracy < self.originalTrainAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 3----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc
            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation

                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 3 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach6(self):
            """ This approach is based on Dixon's method. For each sample, form a match set and then a correct set. The most useful rule in
            the correct set is moved into the final ruleset. In this approach, the most useful rule has the largest accuracy."""

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            retainedClassifiers = []
            self.matchSet = []
            self.correctSet = []

            cons.env.startEvaluationMode()
            cons.env.resetDataRef(True)
            for j in range(cons.env.formatData.numTrainInstances):
                state_class = cons.env.getTrainInstance()
                state = state_class[0]
                classification = state_class[1]

                # Create MatchSet
                for i in range(len(self.pop.popSet)):
                    cl = self.pop.popSet[i]
                    if cl.match(state):
                        self.matchSet.append(i)

                # Create CorrectSet
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if self.pop.popSet[ref].classification == classification:
                        self.correctSet.append(ref)

                # Find the rule with highest accuracy
                highestValue = 0
                highestRef = 0
                for i in range(len(self.correctSet)):
                    ref = self.correctSet[i]
                    product = self.pop.popSet[ref].getAccuracy()
                    if product > highestValue:
                        highestValue = product
                        highestRef = ref

                # If the rule is not already in the final ruleset, move it to the final ruleset
                if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[highestRef])

                    # Move to the next instance
                cons.env.newInstance(True)
                self.matchSet = []
                self.correctSet = []

            cons.env.stopEvaluationMode()

            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach7(self):
            """ Supervise learning version of Dixon's approach. For each sample, form a match set and then a correct set. The most useful rule in
            the correct set is moved into the final ruleset. In this case, the most useful rule has the largest product of accuracy and numerosity."""

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))

            retainedClassifiers = []
            self.matchSet = []
            self.correctSet = []

            cons.env.startEvaluationMode()
            cons.env.resetDataRef(True)
            for j in range(cons.env.formatData.numTrainInstances):
                state_class = cons.env.getTrainInstance()
                state = state_class[0]
                classification = state_class[1]

                # Create MatchSet
                for i in range(len(self.pop.popSet)):
                    cl = self.pop.popSet[i]
                    if cl.match(state):
                        self.matchSet.append(i)

                # Create CorrectSet
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if self.pop.popSet[ref].classification == classification:
                        self.correctSet.append(ref)

                # Find the rule with highest accuracy, numerosity product
                highestValue = 0
                highestRef = 0
                for i in range(len(self.correctSet)):
                    ref = self.correctSet[i]
                    product = self.pop.popSet[ref].getAccuracy() * self.pop.popSet[ref].numerosity
                    if product > highestValue:
                        highestValue = product
                        highestRef = ref

                # If the rule is not already in the final ruleset, move it to the final ruleset
                if highestValue == 0 or self.pop.popSet[highestRef] in retainedClassifiers:
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[highestRef])

                # Move to the next instance
                cons.env.newInstance(True)
                self.matchSet = []
                self.correctSet = []

            cons.env.stopEvaluationMode()

            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach10(self):
            """A method introduced by Kharbat, 2008. Add the rule with the highest entropy to the final compact set. A rule's entropy is
            Defined as (correct matched cases - wrong matched cases)/ number of cases."""

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))

            # Find each rule's entropy
            entropyList = [0.0 for v in range(len(self.pop.popSet))]
            cons.env.startEvaluationMode()
            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                cons.env.resetDataRef(True)
                matchCount = 0
                for j in range(cons.env.formatData.numTrainInstances):
                    cl = self.pop.popSet[i]
                    state = cons.env.getTrainInstance()[0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchCount += 1.0
                    cons.env.newInstance(True)
                entropyList[i] = (matchCount - (
                cons.env.formatData.numTrainInstances - matchCount)) / cons.env.formatData.numTrainInstances
            cons.env.stopEvaluationMode()

            # For each instance in the dataset
            retainedClassifiers = []
            self.matchSet = []
            self.correctSet = []

            cons.env.startEvaluationMode()
            cons.env.resetDataRef(True)
            for j in range(cons.env.formatData.numTrainInstances):
                state_class = cons.env.getTrainInstance()
                state = state_class[0]
                classification = state_class[1]

                # Create MatchSet
                for i in range(len(self.pop.popSet)):
                    cl = self.pop.popSet[i]
                    if cl.match(state):
                        self.matchSet.append(i)

                # Create CorrectSet
                for i in range(len(self.matchSet)):
                    ref = self.matchSet[i]
                    if self.pop.popSet[ref].classification == classification:
                        self.correctSet.append(ref)

                # Find member with highest entropy
                highestValue = -1
                highestRef = 0
                for i in range(len(self.correctSet)):
                    ref = self.correctSet[i]
                    if entropyList[ref] > highestValue:
                        highestValue = entropyList[ref]
                        highestRef = ref

                if highestValue == -1 or self.pop.popSet[highestRef] in retainedClassifiers:
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[highestRef])

                # Move to the next instance
                cons.env.newInstance(True)
                self.matchSet = []
                self.correctSet = []
            cons.env.stopEvaluationMode()

            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach11(self):
            """This approach contains two stages. The first stage is Quick Rule Compaction(QRC). The second stage uses match covering to rank rules(
            Same as stage 3 in approach 1 and 2)."""

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            print("Original Testing Accuracy = " + str(self.originalTestAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            for i in range(len(self.pop.popSet)):
                if self.pop.popSet[i].getAccuracy() <= 0.5 or (
                        self.pop.popSet[i].correctCover == 1 and len(self.pop.popSet[i].specifiedAttList) > 1):
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[i])
            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            completelyGeneralRuleRef = None
            if len(self.pop.popSet) == 0:  # Stop Check
                keepGoing = False
            else:
                keepGoing = True

            # Make the match count list in preparation for stage 2-------------------------------------------------------------------------
            matchCountList = [0.0 for v in range(len(self.pop.popSet))]
            cons.env.startEvaluationMode()
            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                cons.env.resetDataRef(True)
                for j in range(cons.env.formatData.numTrainInstances):  # For each instance in training data
                    cl = self.pop.popSet[i]
                    state = cons.env.getTrainInstance()[0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchCountList[i] += 1
                    cons.env.newInstance(True)
                if len(self.pop.popSet[i].condition) == 0:
                    completelyGeneralRuleRef = i

            cons.env.stopEvaluationMode()
            if completelyGeneralRuleRef != None:  # gets rid of completely general rule.
                del matchCountList[completelyGeneralRuleRef]
                del self.pop.popSet[completelyGeneralRuleRef]

            # ----------------------------------------------------------------------------------------------------------------------------
            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted
            while len(trainingData) > 0 and keepGoing:
                bestRef = None
                bestValue = None
                for i in range(len(matchCountList)):
                    if bestValue == None or bestValue < matchCountList[i]:
                        bestRef = i
                        bestValue = matchCountList[i]

                if bestValue == 0.0 or len(self.pop.popSet) < 1:
                    keepGoing = False
                    continue

                # Update Training Data----------------------------------------------------------------------------------------------------
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[bestRef]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                bestRef])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[bestRef]
                del matchCountList[bestRef]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach14(self):
            """This approach contains two stages. The first stage use Quick Rule Compaction(QRC). The second stage uses fitness to rank the rules
            and guide covering (same as the third stage in approach 2)."""

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            print("Original Testing Accuracy = " + str(self.originalTestAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            for i in range(len(self.pop.popSet)):
                if self.pop.popSet[i].getAccuracy() <= 0.5 or (
                        self.pop.popSet[i].correctCover == 1 and len(self.pop.popSet[i].specifiedAttList) > 1):
                    pass
                else:
                    retainedClassifiers.append(self.pop.popSet[i])
            self.pop.popSet = retainedClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            completelyGeneralRuleRef = None
            if len(self.pop.popSet) == 0:  # Stop check
                keepGoing = False
            else:
                keepGoing = True

            fitnessList = [0.0 for v in range(len(self.pop.popSet))]
            for i in range(len(self.pop.popSet)):
                fitnessList[i] = self.pop.popSet[i].fitness

            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                if len(self.pop.popSet[i].condition) == 0:
                    completelyGeneralRuleRef = i

            if completelyGeneralRuleRef != None:  # gets rid of completely general rule.
                del fitnessList[completelyGeneralRuleRef]
                del self.pop.popSet[completelyGeneralRuleRef]
            # ----------------------------------------------------------------------------------------------------------------------------
            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted
            while len(trainingData) > 0 and keepGoing:
                bestRef = None
                bestValue = None
                for i in range(len(fitnessList)):
                    if bestValue == None or bestValue < fitnessList[i]:
                        bestRef = i
                        bestValue = fitnessList[i]

                if bestValue == 0.0 or len(self.pop.popSet) < 1:
                    keepGoing = False
                    continue

                # Update Training Data----------------------------------------------------------------------------------------------------
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[bestRef]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                bestRef])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[bestRef]
                del fitnessList[bestRef]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach15(self):
            """Only uses fitness to rank the rules and guide covering (same as the third stage in approach 2). Note that this is an old version of
            QCRA. The code is not very efficient and leads to the slow performance. A quicker version is Approach 17. """

            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            print("Original Testing Accuracy = " + str(self.originalTestAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            completelyGeneralRuleRef = None
            if len(self.pop.popSet) == 0:  # Stop check
                keepGoing = False
            else:
                keepGoing = True

            fitnessList = [0.0 for v in range(len(self.pop.popSet))]
            for i in range(len(self.pop.popSet)):
                fitnessList[i] = self.pop.popSet[i].fitness

            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                if len(self.pop.popSet[i].condition) == 0:
                    completelyGeneralRuleRef = i

            if completelyGeneralRuleRef != None:  # gets rid of completely general rule.
                del fitnessList[completelyGeneralRuleRef]
                del self.pop.popSet[completelyGeneralRuleRef]
            # ----------------------------------------------------------------------------------------------------------------------------
            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted
            while len(trainingData) > 0 and keepGoing:
                bestRef = None
                bestValue = None
                for i in range(len(fitnessList)):
                    if bestValue == None or bestValue < fitnessList[i]:
                        bestRef = i
                        bestValue = fitnessList[i]

                if bestValue == 0.0 or len(self.pop.popSet) < 1:
                    keepGoing = False
                    continue

                # Update Training Data----------------------------------------------------------------------------------------------------
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[bestRef]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                bestRef])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[bestRef]
                del fitnessList[bestRef]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach16(self):
            """This approach is a modified version of A15. It uses numerosity to rank rules and guide covering. """
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))
            print("Original Testing Accuracy = " + str(self.originalTestAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            completelyGeneralRuleRef = None
            if len(self.pop.popSet) == 0:  # Stop check
                keepGoing = False
            else:
                keepGoing = True

            numerosityList = [0.0 for v in range(len(self.pop.popSet))]
            for i in range(len(self.pop.popSet)):
                numerosityList[i] = self.pop.popSet[i].numerosity

            for i in range(len(self.pop.popSet)):  # For the population of classifiers
                if len(self.pop.popSet[i].condition) == 0:
                    completelyGeneralRuleRef = i

            if completelyGeneralRuleRef != None:  # gets rid of completely general rule.
                del numerosityList[completelyGeneralRuleRef]
                del self.pop.popSet[completelyGeneralRuleRef]
            # ----------------------------------------------------------------------------------------------------------------------------
            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted
            while len(trainingData) > 0 and keepGoing:
                bestRef = None
                bestValue = None
                for i in range(len(numerosityList)):
                    if bestValue == None or bestValue < numerosityList[i]:
                        bestRef = i
                        bestValue = numerosityList[i]

                if bestValue == 0.0 or len(self.pop.popSet) < 1:
                    keepGoing = False
                    continue

                # Update Training Data----------------------------------------------------------------------------------------------------
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[bestRef]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                bestRef])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[bestRef]
                del numerosityList[bestRef]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def Approach18(self):
            """A transitional version between Fu's approach and QCRA. The first two stages are the same as Fu's. The third stage uses fitness to
            rank rules and guide covering. """

            # Order Classifier Set---------------------------------------------------------------------------------------------------------
            lastGood_popSet = sorted(self.pop.popSet, key=self.numerositySort)
            self.pop.popSet = lastGood_popSet[:]
            print("Starting number of classifiers = " + str(len(self.pop.popSet)))
            print("Original Training Accuracy = " + str(self.originalTrainAcc))

            # STAGE 1----------------------------------------------------------------------------------------------------------------------
            keepGoing = True
            while keepGoing:
                del self.pop.popSet[0]  # Remove next classifier
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)  # DEBUG
                if newAccuracy < self.originalTrainAcc:
                    keepGoing = False
                    self.pop.popSet = lastGood_popSet[:]
                else:
                    lastGood_popSet = self.pop.popSet[:]
                if len(self.pop.popSet) == 0:
                    keepGoing = False
            print("STAGE 1 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 2----------------------------------------------------------------------------------------------------------------------
            retainedClassifiers = []
            RefAccuracy = self.originalTrainAcc
            for i in range(len(self.pop.popSet)):
                heldClassifier = self.pop.popSet[0]
                del self.pop.popSet[0]
                newAccuracy = self.performanceEvaluation(True)  # Perform classifier set training accuracy evaluation
                print(newAccuracy)
                if newAccuracy < RefAccuracy:
                    retainedClassifiers.append(heldClassifier)
                    RefAccuracy = newAccuracy

            self.pop.popSet = retainedClassifiers
            print("STAGE 2 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

            # STAGE 3----------------------------------------------------------------------------------------------------------------------
            finalClassifiers = []
            if len(self.pop.popSet) == 0:  # Stop check
                keepGoing = False
            else:
                keepGoing = True

            lastGood_popSet = sorted(self.pop.popSet, key=self.accuracySort, reverse=True)
            self.pop.popSet = lastGood_popSet[:]

            tempEnv = copy.deepcopy(cons.env)
            trainingData = tempEnv.formatData.trainFormatted

            while len(trainingData) > 0 and keepGoing:
                newTrainSet = []
                matchedData = 0
                for w in range(len(trainingData)):
                    cl = self.pop.popSet[0]
                    state = trainingData[w][0]
                    doesMatch = cl.match(state)
                    if doesMatch:
                        matchedData += 1
                    else:
                        newTrainSet.append(trainingData[w])
                if matchedData > 0:
                    finalClassifiers.append(self.pop.popSet[
                                                0])  # Add best classifier to final list - only do this if there are any remaining matching data instances for this rule!
                # Update classifier list and training set list
                trainingData = newTrainSet
                del self.pop.popSet[0]
                if len(self.pop.popSet) == 0:
                    keepGoing = False

            self.pop.popSet = finalClassifiers
            print("STAGE 3 Ended: Classifiers Remaining = " + str(len(self.pop.popSet)))

        ############################################################################################################################################################################################
        def performanceEvaluation(self, isTrain):
            """ Performs Training or Testing Evaluation"""

            cons.env.startEvaluationMode()
            cons.env.resetDataRef(isTrain)
            if isTrain:
                myType = "TRAINING"
            else:
                myType = "VALIDATION"
            noMatch = 0  # How often does the population fail to have a classifier that matches an instance in the data.
            tie = 0  # How often can the algorithm not make a decision between classes due to a tie.
            cons.env.resetDataRef(isTrain)  # Go to the first instance in dataset
            phenotypeList = cons.env.formatData.phenotypeList
            # ----------------------------------------------
            classAccDict = {}
            for each in phenotypeList:
                classAccDict[each] = ClassAccuracy()
            # ----------------------------------------------
            if isTrain:
                instances = cons.env.formatData.numTrainInstances
            else:
                instances = cons.env.formatData.numTestInstances
            # ----------------------------------------------------------------------------------------------
            for inst in range(instances):
                if isTrain:
                    state_phenotype = cons.env.getTrainInstance()
                else:
                    state_phenotype = cons.env.getTestInstance()
                # -----------------------------------------------------------------------------
                self.pop.makeEvalMatchSet(state_phenotype[0])
                prediction = Prediction(self.pop)
                phenotypeSelection = prediction.getDecision()
                # -----------------------------------------------------------------------------

                if phenotypeSelection == None:
                    noMatch += 1
                elif phenotypeSelection == 'Tie':
                    tie += 1
                else:  # Instances which failed to be covered are excluded from the accuracy calculation
                    for each in phenotypeList:
                        thisIsMe = False
                        accuratePhenotype = False
                        truePhenotype = state_phenotype[1]
                        if each == truePhenotype:
                            thisIsMe = True
                        if phenotypeSelection == truePhenotype:
                            accuratePhenotype = True
                        classAccDict[each].updateAccuracy(thisIsMe, accuratePhenotype)

                cons.env.newInstance(isTrain)  # next instance
                self.pop.clearSets()
                # ----------------------------------------------------------------------------------------------
            # Calculate Standard Accuracy--------------------------------------------
            instancesCorrectlyClassified = classAccDict[phenotypeList[0]].T_myClass + classAccDict[phenotypeList[0]].T_otherClass
            instancesIncorrectlyClassified = classAccDict[phenotypeList[0]].F_myClass + classAccDict[phenotypeList[0]].F_otherClass
            standardAccuracy = float(instancesCorrectlyClassified) / float(instancesCorrectlyClassified + instancesIncorrectlyClassified)

            # Calculate Balanced Accuracy---------------------------------------------
            T_mySum = 0
            T_otherSum = 0
            F_mySum = 0
            F_otherSum = 0
            for each in phenotypeList:
                T_mySum += classAccDict[each].T_myClass
                T_otherSum += classAccDict[each].T_otherClass
                F_mySum += classAccDict[each].F_myClass
                F_otherSum += classAccDict[each].F_otherClass
            balancedAccuracy = ((0.5 * T_mySum / (float(T_mySum + F_otherSum)) + 0.5 * T_otherSum / (
            float(T_otherSum + F_mySum))))  # BalancedAccuracy = (Specificity + Sensitivity)/2

            # Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
            predictionFail = float(noMatch) / float(instances)
            predictionTies = float(tie) / float(instances)
            instanceCoverage = 1.0 - predictionFail
            predictionMade = 1.0 - (predictionFail + predictionTies)

            adjustedStandardAccuracy = (standardAccuracy * predictionMade) + (
            (1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
            adjustedBalancedAccuracy = (balancedAccuracy * predictionMade) + (
            (1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))

            # Adjusted Balanced Accuracy is calculated such that instances that did not match have a consistent probability of being correctly classified in the reported accuracy.

            """
            print("-----------------------------------------------")
            print(str(myType) + " Accuracy Results for Model " + str(exp) + ":-------------")
            print("Instance Coverage = " + str(instanceCoverage * 100.0) + '%')
            print("Prediction Ties = " + str(predictionTies * 100.0) + '%')
            print(str(instancesCorrectlyClassified) + ' out of ' + str(
                instances) + ' instances covered and correctly classified.')
            print("Standard Accuracy (Adjusted) = " + str(adjustedStandardAccuracy))
            print("Balanced Accuracy (Adjusted) = " + str(adjustedBalancedAccuracy))
            """

            # Balanced and Standard Accuracies will only be the same when there are equal instances representative of each phenotype AND there is 100% covering.
            cons.env.stopEvaluationMode()
            return adjustedBalancedAccuracy

        def numerositySort(self, cl):
            """ Sorts from smallest numerosity to largest """
            return cl.numerosity

        def accuracySort(self, cl):
            return cl.getAccuracy()


########################################################################################################################

    #Initialize the 'Timer' module which tracks the run time of algorithm and it's different components.
    timer = Timer()
    cons.referenceTimer(timer)

    #Initialize the 'Environment' module which manages the data presented to the algorithm.  While LCS learns iteratively (one inistance at a time
    env = Offline_Environment()
    cons.referenceEnv(env) #Passes the environment to 'Constants' (cons) so that it can be easily accessed from anywhere within the code.
    cons.parseIterations() #Identify the maximum number of learning iterations as well as evaluation checkpoints.
    alg = eLCS()
    return(alg.perfreport)


