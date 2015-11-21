# svm_bench.py
import logging
import my_io
import classification_baseline
from sklearn import svm
import numpy as np
from sklearn.metrics import classification
from numpy import linalg as LA

my_io.setUp('./biological_response/')

my_io.startLog(__name__)
logger = logging.getLogger(__name__)

y,X,trainData,testData = my_io.readCsv()
portion = 0.2
seed = 1
X_test, X_train, y_train, y_test = classification_baseline.splitData(X,y,portion,seed)

logger.info('init svm classifier')
svc = svm.SVC(probability = True)
logger.info('fitting svc')
svc.fit(X_train, y_train)
logger.info('start predict')
predict_probs = svc.predict_proba(X_test)

predict = my_io.toZeroOne(predict_probs)
# error = classification.zero_one_loss(y_test, predict)
loss = np.subtract(predict,y_test)

error = LA.norm(loss)
logger.info('zero one loss %f',error)