import os.path

class Classifier:

    def __init__(self, dataInfo, a=None, b=None, c=None, d=None):
        # Major Parameters --------------------------------------------------
        self.specifiedAttList = []  # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.condition = []  # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.phenotype = None  # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous

        self.fitness = 0.0  # Classifier fitness - initialized to a constant initial fitness value
        self.accuracy = 0.0  # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.numerosity = 1  # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
        self.aveMatchSetSize = None  # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.deletionVote = None  # The current deletion weight for this classifier.

        # Experience Management ---------------------------------------------
        self.timeStampGA = None  # Time since rule last in a correct set.
        self.initTimeStamp = None  # Iteration in which the rule first appeared.

        # Classifier Accuracy Tracking -------------------------------------
        self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0  # The total number of times this classifier was in a correct set

        self.dataInfo = dataInfo
        self.labelMissingData = 'NA'

        if isinstance(c, list):  # note subtle new change
            self.classifierCovering(a, b, c, d)
        elif isinstance(a, Classifier):
            self.classifierCopy(a, b)
        elif isinstance(a, list) and b == None:
            self.rebootClassifier(a)
        else:
            print("Classifier: Error building classifier.")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def match(self, state):
        """ Returns if the classifier matches in the current situation. """
        for i in range(len(self.condition)):
            attInfo = self.dataInfo[1]
            attributeInfo = attInfo[self.specifiedAttList[i]]
            # -------------------------------------------------------
            # CONTINUOUS ATTRIBUTE
            # -------------------------------------------------------
            if attributeInfo[0]:
                instanceValue = state[self.specifiedAttList[i]]
                if float(self.condition[i][0]) < instanceValue < float(self.condition[i][1]) or instanceValue == self.labelMissingData:
                    pass
                else:
                    return False
                    # -------------------------------------------------------
            # DISCRETE ATTRIBUTE
            # -------------------------------------------------------
            else:
                stateRep = state[self.specifiedAttList[i]]
                if stateRep == self.condition[i] or stateRep == self.labelMissingData:
                    pass
                else:
                    return False
        return True

    def reportClassifier(self):
        """  Transforms the rule representation used to a more standard readable format. """
        numAttributes = self.dataInfo[0]
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
        numAttributes = self.dataInfo[0]
        attInfo = self.dataInfo[1]
        discretePhenotype = self.dataInfo[2]
        for attRef in range(numAttributes):
            attributeInfo = attInfo[attRef]
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
        specificity = len(self.condition) / float(numAttributes)

        if discretePhenotype:
            classifierString += str(self.phenotype) + "\t"
        else:
            classifierString += str(self.phenotype[0]) + ';' + str(self.phenotype[1]) + "\t"

        # print(self.deletionVote)    does this not occur until population is full???
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifierString += str("%.3f" % self.fitness) + "\t" + str("%.3f" % self.accuracy) + "\t" + str(
            "%d" % self.numerosity) + "\t" + str("%.2f" % self.aveMatchSetSize) + "\t" + str(
            "%d" % self.timeStampGA) + "\t" + str("%d" % self.initTimeStamp) + "\t" + str("%.2f" % specificity) + "\t"
        classifierString += "\t" + str("%d" % self.correctCount) + "\t" + str("%d" % self.matchCount) + "\n"

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return classifierString

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def rebootClassifier(self, classifierList):
        """ Rebuilds a saved classifier as part of the population Reboot """
        numAttributes = self.dataInfo[0]
        attInfo = self.dataInfo[1]
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
        if self.dataInfo[2]:
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
        # self.deletionVote = float(classifierList[numAttributes + 8])
        self.correctCount = int(classifierList[numAttributes + 9])
        self.matchCount = int(classifierList[numAttributes + 10])


class ClassifierSet:

    def __init__(self, dataManage,  a=None):
        """ Overloaded initialization: Handles creation of a new population or a rebooted population (i.e. a previously saved population). """
        # Major Parameters
        self.popSet = []  # List of classifiers/rules
        self.matchSet = []  # List of references to rules in population that match
        self.microPopSize = 0  # Tracks the current micro population size

        # Evaluation Parameters-------------------------------
        self.aveGenerality = 0.0
        self.expRules = 0.0
        self.attributeSpecList = []
        self.attributeAccList = []
        self.avePhenotypeRange = 0.0
        self.numAttributes = dataManage.numAttributes
        self.attributeInfo = dataManage.attributeInfo
        self.discretePhenotype = dataManage.discretePhenotype
        self.dataInfo = [self.numAttributes, self.attributeInfo, self.discretePhenotype]

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
    def rebootPop(self, remakeFile):
        """ Remakes a previously evolved population from a saved text file. """
        #print("Rebooting the following model: " + str(os.path.basename(remakeFile)))
        # *******************Initial file handling**********************************************************
        try:
            datasetList = []
            # read_path = cons.popRebootPath
            file_name = remakeFile
            f = open(file_name, 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open ', os.path.basename(remakeFile))
            raise
        else:
            self.headerList = f.readline().rstrip('\n').split('\t')  # strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()

            # **************************************************************************************************
        for each in datasetList:
            cl = Classifier(self.dataInfo, each)
            self.popSet.append(cl)
            self.microPopSize += 1
        #print("Rebooted Rule Population has " + str(len(self.popSet)) + " Macro Pop Size.")

