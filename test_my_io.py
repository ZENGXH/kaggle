# test my_io.py

import my_io
import numpy as np
import logging
# test setUp
my_io.setUp('digit/')

# test read data
my_io.readCsv()

a = [1,2,3]
my_io.writeCsv(a)


my_io.startLog(__name__)
logger = logging.getLogger(__name__)
logger.info('pass test :) ') 