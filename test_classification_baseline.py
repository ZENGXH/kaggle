# test_classification_baseline.py

import numpy as np
import classification_baseline
import logging
from my_io import startLog
startLog(__name__)
logger = logging.getLogger(__name__)

X = np.array([[1,2,3,54,64],
			[1,23,32,54,64],
			[1,233,2,54,64],
			[1,263,3,54,64],
			[1,37,32,54,64],
			[10,22,3,4,6]])
y = np.array([[2],
			[42],
			[231],
			[32],
			[22],
			[21]])

portion = 0.2

seed = 1

X_test, X_train, y_train, y_test = classification_baseline.splitData(X,y,portion,seed)

print 'test:', X_test
print y_test

print 'train', X_train
print y_train 

# assert %size of result == %expectation 
logger.info('pass test :)')

