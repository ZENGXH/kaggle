# test kmeans.py
import logging
import kmeans
import numpy as np
import my_io
import gmm
import random
# import 
# initialize
my_io.startLog(__name__)
logger = logging.getLogger(__name__)


# for small size of data, use genfromtxt which return array
# for large size, use csv.reader return str array
X = np.genfromtxt('./faithful.txt')
logger.info('import data, in shape %s',str(np.shape(X)))



# for k means, initilize k
k = 4
print np.shape(X)
logger.info('k is %s',str(k))
# kmeans.kmeansClustering(X,k)
gmm.gmmClustering(X,2)