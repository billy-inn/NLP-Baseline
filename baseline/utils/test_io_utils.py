import config
import logging
from . import io_utils

def test_read_snli():
    logging.info("=" * 50)
    logging.info("Testing read_snli...")
    logging.info("Reading Validation Set...")
    sent1, sent2, labels = io_utils.read_snli(config.SNLI_VALID)
    logging.info("Length of the output %d" % len(labels))
    logging.info("Elements in the first entry:")
    logging.info(str(sent1[0]))
    logging.info(str(sent2[0]))
    logging.info(str(labels[0]))
    logging.info("=" * 50)

def test_io_utils():
    logging.info("Testing io_utils...")
    test_read_snli()
    logging.info("Testing completed!")
