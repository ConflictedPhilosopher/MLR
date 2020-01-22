SEED_NUMBER = 161
NUMBER_OF_FOLDS = 1                #k-fold cross-validation model training
DO_AVERAGING = False               #reoprt averages performance
NO_PARALLEL_JOBS = 10              #number of threads to perform parallel model training


DATA_HEADER = "flags"
NO_ATTRIBUTES = 19
VALID_DATA_HEADER = DATA_HEADER + "-test"
TRAIN_DATA_HEADER = DATA_HEADER + "-train"
DATA_FOLDER = "D:\Datasets"
RUN_RESULT_PATH = "Run_results_MLRBC"


NO_TRAIN_ITERATION = [10000]
POP_SIZE = [5000]
P_HASH = [0.1]
THETA_GA = [5] #[10,15,20,25,30,35,40,45,50,55,60]
NU = [50]
NO_EXPERIMENTS_AVERAGING = 1      #repeat the training to reduce the variance
DISTRIBUTED_MATCHING_TH = 2600000   
REDUCE_ATTRIBUTE = None              #feature selection parameter (1, <1, None). 1: considers feature-list.txt
                                  # <1: takes a portion of feature-list.txt, None: no feature selection
REBOOT_MODEL = 0                  #reboot a previously trained model from the file
DOWN_SAMPLE_RATIO = 1             #down_sampling ratio. For faster prototype evaluations
PREDICTION_METHOD = 'agg'         #set to 'agg' for combined prediction. set to 'max' for highest vote
THRESHOLD = 'onethreshold'        #'onethreshold', 'rcut', 'pcut'
THETA_THRESHOLD = 0.8             #threshold for the one-threshold bipartition method
RCUT_T = 3                        #threshold for the rank-cut bipartition method. None or a decimal value
PCUT_W = 0.5                      #threshold for the class-specific score-cut bipartition method. None or from [0,1].
ADAPT_THETA_GA = False            #adaptive GA thershold selection for imbalanced data. True or False
PLOT_SETTING = [0, 1, 1, 0, 0]    #population sizes, accuracy, Hloss, generality, TP & TN
CLUSTERING_MODE = 'local'         #'local': clustering in [M]. 'global': clustering on Dataset
CLUSTERING_METHOD = 'graph'       #label clsutering method. set to 'graph' or 'density'

# REF_CARDINALITY = None          #set to 'None' for no density modification