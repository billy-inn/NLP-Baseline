
# ---------------------- PATH ------------------------------
ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
LOG_PATH = "%s/log" % ROOT_PATH
CHECKPOINT_PATH = "%s/checkpoint" % ROOT_PATH

SNLI_PATH = "%s/snli_1.0" % DATA_PATH

# ---------------------- DATA -----------------------------

SNLI_TRAIN = "%s/snli_1.0_train.jsonl" % SNLI_PATH
SNLI_VALID = "%s/snli_1.0_dev.jsonl" % SNLI_PATH
SNLI_TEST = "%s/snli_1.0_test.jsonl" % SNLI_PATH

EMBEDDING_DATA = "%s/glove.840B.300d.txt" % DATA_PATH

# ---------------------- PARAM ----------------------------

SNLI_MAX_LENGTH = 50
