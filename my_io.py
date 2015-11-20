"""this file is to read the train.csv and test csv file
    from kaggle, in general, read from the second row,
    for train file, the first column is the output data
    and the others are the inputdata
    y: [N, 1]
    X: [N, D]
    testData: [N2, D]
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy import genfromtxt, savetxt


def readCsv(trainFile, testFile):
    # create the training & test sets, skipping the header row with [1:]
    print 'start'
    # read csv file from second row

    trainData = genfromtxt(open(trainFile,'r'), delimiter=',', dtype='f8')[1:] 
    print('size of training data (#training_datapoints, #features/dimension + 1): ',
            np.shape(trainData))
    # info:
    # shape (#training_datapoints, #features/dimension + 1), 
    # the first col is the output value of the training data
    # ie, each rows is inform: (output_i, input_i'(row vector))

    testData = genfromtxt(open(trainFile,'r'), delimiter=',', dtype='f8')[1:]
    print 'size of test data (N,1)', np.shape(testData)
    # info:
    # shape (#test_datapoints, #features/dimension) ie [n,1]

    # target, ie, y  is the first col of the dataset
    # for each row in trainData, extract the first element and form a list
    y = [x[0] for x in trainData] # N*1
    # info:
    # shape (#training_datapoints, )

    # inputdata, ie X is the second row to the last of the dataset
    # for each row in trainData, extract the second to the end element and form a list
    X = [row[1:] for row in trainData]  # N*D
    print 'return y as array, X as array [N,D], trainData, testData'
    return y,X,trainData,testData
	
"""write submission file
	input should be list
"""
def write_delimited_file(file_path, data,header=None, delimiter=","):
    f_out = open(file_path,"w")
    if header is not None:
        f_out.write(delimiter.join(header) + "\n")
    for line in data:
        if isinstance(line, str):
            f_out.write(line + "\n")
        else:
            f_out.write(delimiter.join(line) + "\n")
    f_out.close()


def writeCsv(result,saveFile):
	write_delimited_file(saveFile, result)
	print 'save result as ' + saveFile
	
