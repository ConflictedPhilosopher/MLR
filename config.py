SEED_NUMBER = 161
NUMBER_OF_FOLDS = 1
NO_EXPERIMENTS_AVERAGING = 2
DO_AVERAGING = True
NO_PARALLEL_JOBS = 10


DATA_HEADER = "yeast"
NO_ATTRIBUTES = 103
VALID_DATA_HEADER = DATA_HEADER + "-test"
TRAIN_DATA_HEADER = DATA_HEADER + "-train"
DATA_FOLDER = "C:\Datasets"
RUN_RESULT_PATH = "Run_results_MLRBC"

NO_TRAIN_ITERATION = 100
POP_SIZE = 100
DISTRIBUTED_MATCHING_TH = 50
REDUCE_ATTRIBUTE = 0.3
REF_CARDINALITY = None            # set to 'None' for no density modification
REBOOT_MODEL = 0
DOWN_SAMPLE_RATIO = 1
PREDICTION_METHOD = 'max'         # set to 'agg' for combined prediction
THRESHOLD = 'rcut'        # onethreshold, rcut
ADAPT_THETA_GA = False
PLOT_SETTING = [0, 1, 1, 0, 0]    # population sizes, accuracy, Hloss, generality, TP & TN


