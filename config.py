SEED_NUMBER = 161
NUMBER_OF_FOLDS = 1
NO_EXPERIMENTS_AVERAGING = 1
DO_AVERAGING = True
NO_PARALLEL_JOBS = 10


DATA_HEADER = "yeast"
NO_ATTRIBUTES = 103
VALID_DATA_HEADER = DATA_HEADER + "-test"
TRAIN_DATA_HEADER = DATA_HEADER + "-train"
DATA_FOLDER = "C:\Datasets"
RUN_RESULT_PATH = "Run_results_MLRBC"

NO_TRAIN_ITERATION = 500
POP_SIZE = 50
DISTRIBUTED_MATCHING_TH = 6000
REDUCE_ATTRIBUTE = 0.1
REF_CARDINALITY = None            # set to 'None' for no density modification
DOWN_SAMPLE_RATIO = 1
PREDICTION_METHOD = 'agg'         # set to 'agg' for combined prediction
THRESHOLD = 'pcut'        # onethreshold, rcut
ADAPT_THETA_GA = False
PLOT_SETTING = [0, 1, 1, 0, 0]    # population sizes, accuracy, Hloss, generality, TP & TN


