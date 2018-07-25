import config
import logging
from . import io_utils, data_utils

def test_data_utils():
    logging.info("Testing data_utils...")
    logging.info("=" * 50)
    logging.info("Reading Training Set...")
    sent1, sent2, labels = io_utils.read_snli(config.SNLI_TRAIN)
    logging.info("Number of valid training examples: %d" % len(labels))
    max_len = 0
    for s in sent1 + sent2:
        max_len = max(max_len, len(s))
    logging.info("Maximum length of the sentences: %d" % max_len)
    label_dict = data_utils.create_label_dict(labels)
    logging.info("Created label dict:")
    logging.info(str(label_dict))
    logging.info("=" * 50)
    logging.info("Testing completed!")
