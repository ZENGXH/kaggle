from sklearn.metrics import classification
import numpy as np
import math
import my_io
import logging
from my_io import startLog
        
def splitData(X,y,portion,seed):

	startLog(__name__)
	logger = logging.getLogger(__name__)
	records = {'portion':portion, 'seed':seed}
	logger.info('split data into train and test %s',records)

	if not isinstance(X, np.ndarray):
		logger.debug('X is not a nparray, converted')
		X = np.array(X)
		y = np.array(y)
	(rows, cols) = np.shape(X)
	size_data = rows
	index = np.random.permutation(size_data)

	Ntr = math.floor(portion*size_data)

	idx_test = index[0:Ntr]
	idx_train = index[Ntr:]
	X_test = X[idx_test,:]
	X_train = X[idx_train,:]

	y_test = y[idx_test]
	y_train = y[idx_train]

	logger.info('split done check split')

	# check split
	(rows_test, cols_test) = np.shape(X_test)
	(rows_train, cols_train) = np.shape(X_train)
	if rows_train < rows_train:
		logger.warning('test set larger than training set')

	logger.info('size of X_test is %d and X_train %d, #fts %d',rows_test,rows_test,cols_train)
	return X_test, X_train, y_train, y_test


# loss = classification.zero_one_loss(y_true, y_pred)