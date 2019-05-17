SEED_NUMBER = 161
NUMBER_OF_FOLDS = 1
NO_EXPERIMENTS_AVERAGING = 10
DO_AVERAGING = False
NO_PARALLEL_JOBS = 10


DATA_HEADER = "CAL500"
NO_ATTRIBUTES = 68
VALID_DATA_HEADER = DATA_HEADER + "-test"
TRAIN_DATA_HEADER = DATA_HEADER + "-train"
DATA_FOLDER = "C:\Datasets"
RUN_RESULT_PATH = "Run_results_MLRBC"

NO_TRAIN_ITERATION = 80000
POP_SIZE = 2000
P_HASH = 0.9
DISTRIBUTED_MATCHING_TH = 2600000
REDUCE_ATTRIBUTE = 1
REF_CARDINALITY = None            # set to 'None' for no density modification
REBOOT_MODEL = 0
DOWN_SAMPLE_RATIO = 0.5
PREDICTION_METHOD = 'max'         # set to 'agg' for combined prediction
THRESHOLD = 'rcut'        # onethreshold, rcut, pcut
THETA_THRESHOLD = 0.55
RCUT_T = 3          # None or a decimal value
PCUT_W = 0.5
ADAPT_THETA_GA = False
PLOT_SETTING = [0, 1, 1, 0, 0]    # population sizes, accuracy, Hloss, generality, TP & TN