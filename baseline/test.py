from utils.test_io_utils import test_io_utils
from utils.test_data_utils import test_data_utils
from models.test_esim import test_esim
from utils.logging_utils import _set_basic_logging

def main():
    # test_io_utils()
    # test_data_utils()
    test_esim()

if __name__ == "__main__":
    _set_basic_logging()
    main()
