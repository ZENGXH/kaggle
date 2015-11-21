# test my_io.py

import my_io
import numpy as np
import logging

# test setUp
my_io.setUp('digit/')
startLog(__name__)
logger = logging.getLogger(__name__)

"""# test read data
my_io.readCsv()
a = [1,2,3]
my_io.writeCsv(a)
my_io.startLog(__name__)
logger = logging.getLogger(__name__)
"""

a = [['1','2.2','3.3'],['3.1','2.3','2']]
b = a[0]

print my_io.toFloat(a)
print my_io.toZeroOne(b)

logger.info('pass test :) ') 

