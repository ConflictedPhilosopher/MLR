SEED_NUMBER = 161
NUMBER_OF_FOLDS = 1
NO_EXPERIMENTS_AVERAGING = 10
DO_AVERAGING = False
NO_PARALLEL_JOBS = 10


DATA_HEADER = "yeast"
NO_ATTRIBUTES = 103
VALID_DATA_HEADER = DATA_HEADER + "-test"
TRAIN_DATA_HEADER = DATA_HEADER + "-train"
DATA_FOLDER = "C:\Datasets"
RUN_RESULT_PATH = "Run_results_MLRBC"

NO_TRAIN_ITERATION = 120000
POP_SIZE = 12000
DISTRIBUTED_MATCHING_TH = 26000
REDUCE_ATTRIBUTE = 0.3
REF_CARDINALITY = None            # set to 'None' for no density modification
REBOOT_MODEL = 1
DOWN_SAMPLE_RATIO = 1
PREDICTION_METHOD = 'agg'         # set to 'agg' for combined prediction
THRESHOLD = 'onethreshold'        # onethreshold, rcut, pcut
THETA_THRESHOLD = 0.47
RCUT_T = 4          # None or a decimal value
PCUT_W = 0.5
ADAPT_THETA_GA = False
PLOT_SETTING = [0, 1, 1, 0, 0]    # population sizes, accuracy, Hloss, generality, TP & TN