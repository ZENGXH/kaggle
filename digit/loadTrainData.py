# tutorial from 
# http://blog.csdn.net/u012162613/article/details/41929171

import numpy
import csv


# the data read from csv file is in string fornumpy.mat
# therefore we need transform them into Int
def toInt(array):
	array = numpy.mat(array)
	m,n = numpy.shape(array)
	newArray = numpy.zeros((m, n))
	for i in xrange(m):
		for j in xrange(n):
			newArray[i, j] = int(array[i,j])

	return newArray

# convert the image value from 0-255 into binary data
def normalizing(array):
	m, n = numpy.shape(array)

	for i in xrange(m):
		for j in xrange(n):
			if array[i, j] != 0:
				array[i, j] = 1
	return array

def loadTrainData():
	print 'loadTraindata..'
	l = []
	with open('train.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line) # 42001 * 685
	l.remove(l[0])
	l = numpy.array(l)
	label = l[:,0]
	data = l[:,1:]

	return normalizing(toInt(data)),toInt(label)

def loadTestData():
	print 'loadTestData...'
	l = []
	with open('test.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line)

	l.remove(l[0])
	data = numpy.array(1)

	return normalizing(toInt(data))


def loadTestResult():
	print('loadTestResult..')
	l = []
	with open('knn_benchmark.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line)
	l.remove(l[0])
	label = numpy.array(1)

	return toInt(label[:, 1]) # label 28000 * 1


def saveResult(result, csvName):

	with open(csvNAME, 'wb') as myFile:
		myWritter = csv.writer(myFile)
		for i in result:
			tmp = []
			tmp.append(i)
			myWritter.writerow(tmp)
	print 'result save as' + myFile

from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData, trainLabel, testData):
	knnClf = KNeighborsClassifier(n_neighbors = 10)
	knnClf.fit(trainData, numpy.ravel(trainLabel))

	testLabel = knnClf.predict(testData)

	saveResult(testData, 'sklearn_knn_result.csv')
	return testLabel

def classify(inX, dataSet, labels, k):
	inX = numpy.mat(inX)
	dataSet = numpy.mat(dataSet)
	labels = numpy.mat(labels)

	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet

def digitReconition():
	trainData, trainLabel = loadTrainData()
	testData = loadTestData()

	result = knnClassify(trainData, trainLabel, testData)

	resultGiven = loadTestResult()
	m, n = numpy.shape(testData)
	diff = 0
	for i in xrange(m):
		if result[i] != resultGiven[0, i]:
			diff += 1
	print diff

def main():
	print(" - Start. ")
	digitReconition()
	print(" - finished ")

if __name__ == '__main__':
	main()