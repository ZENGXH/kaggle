"""some io stuff and data format converter

"""
import logging
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy import genfromtxt, savetxt
import ConfigParser
import datetime
import os
import csv



""" python debugger
    import pdb
    pdb.set_trace()

"""
def startLog(loggerName):
    """useage:
        from my_io import startLog
        startLog(__name__)
        log.info('msg %d %s', int, str)
        logger.debug('%s iteration, item=%s', i, item)

        user = db.read_user(user_id)
        if user is None:
        logger.error('Cannot find user with user_id=%s', user_id)

        try:
            open('/path/to/does/not/exist', 'rb')
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception, e:
            logger.error('Failed to open file', exc_info=True)

        seek to 
        http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python
        if you want to use yaml or jason config file format

    """
    # get logFile from confiure.ini
    config = ConfigParser.ConfigParser()
    config.read('logging_config.ini')
    logFile = config.get('logSetting','logFile')

    # FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    #logging.basicConfig(filename= logFile,level=logging.DEBUG, 
    #                    format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')
    logger = logging.getLogger(loggerName)

    if len(logger.handlers)==0:
        fh = logging.FileHandler(logFile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    logger.setLevel(logging.DEBUG)
 
def setUp(path):
    currentTime = datetime.datetime.now()

    config = ConfigParser.ConfigParser()
    if not os.path.exists('logging_config.ini'):
        open('logging_config.ini','w').close()

    config.read('logging_config.ini')

    """write ini file based on the path info

    """
    # initialze the logFile in configure file    
    logFile = path+'log.txt'
    if not config.has_section('logSetting'):                                
           config.add_section('logSetting')   
    config.set('logSetting','logFile',logFile)

    trainFile = path + 'train.csv'
    testFile = path + 'test.csv'
    appen = (str(currentTime.month)+
            str(currentTime.day) + 
            str(currentTime.hour) + 
            str(currentTime.minute))
    saveFile = path + 'result'+appen+'.csv'
    if not config.has_section('data'):                                
           config.add_section('data')
    config.set('data','trainFile',trainFile)
    config.set('data','testFile',testFile)
    config.set('data','saveFile',saveFile)
    config.write(open('logging_config.ini', "w"))
    # config write done

    # start recording

    startLog(__name__)
    logger = logging.getLogger(__name__)
    logger.info('logger first setup %s',logFile)

#    readCsv(trainFile, testFile)


# the data read from csv file is in string fornumpy.mat
# therefore we need transform them into Int
def toFloat(oldArray):
    startLog(__name__)
    logger = logging.getLogger(__name__)
    logger.info('check type of array: %s',type(oldArray))
    # initialize
    newArray = oldArray

    if len(np.shape(oldArray)) == 1:
        logger.info('1d array')
        if not isinstance(oldArray[0], float):
            logger.debug('array not float, converting..')
            newArray = np.array(oldArray, dtype='f8')
    elif len(np.shape(oldArray)) == 2:
        logger.info('2d array')
        if not isinstance(oldArray[0][0], float):
            logger.debug('array not float, converting..')
            newArray = np.array(oldArray, dtype='f8')        
    else:
        logger.warning('array dimension more than 2, convert anyway')
        newArray = np.array(oldArray, dtype='f8')   
    
    """array = np.mat(array)
    
    newArray = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = int(array[i,j])
    """
    logger.info('toFloat done')
    return newArray

def getExtrme(oldArray):

    startLog(__name__)
    logger = logging.getLogger(__name__)
    logger.info('maximum of probabiliy %f', np.amax(oldArray))
    logger.info('minimum of probabiliy %f', np.amin(oldArray))

# the data read from csv file is in string fornumpy.mat
# therefore we need transform them into Int
def toZeroOne(oldArray):
    #import pdb
    #pdb.set_trace()

    startLog(__name__)
    logger = logging.getLogger(__name__)
    logger.info('check type of array: %s',type(oldArray))
    if isinstance(oldArray, list):
        logger.info('convert to  array')
        oldArray = np.array(oldArray)

    # initialize
    # newArray = oldArray
    if len(np.shape(oldArray)) == 1:
        logger.info('1d array')
        if not isinstance(oldArray[0], float):
            logger.debug('array not float, converting to float first')
            oldArray = toFloat(oldArray)
        newArray = [1 if x > 0.5 else 0 for x in oldArray]
    else:
        logger.warning('array dimension more than 1'+
                        'use the second col')
        oldArray = oldArray[:,1]
        if not isinstance(oldArray[0], float):
            logger.debug('array not float, converting to float first')
            oldArray = toFloat(oldArray)
        newArray = [1 if x > 0.5 else 0 for x in oldArray]        
        #newArray = np.array(oldArray, dypt='f8')   
    getExtrme(oldArray)
    """array = np.mat(array)
    
    newArray = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = int(array[i,j])
    """
    logger.info('toZeroOne done')
    return newArray

def readCsv():
    """this is to read the train.csv and test csv file
        from kaggle, in general, read from the second row,
        for train file, the first column is the output data
        and the others are the inputdata
        y: [N, 1]
        X: [N, D]
        testData: [N2, D]
    """
    # init:
    # get current state from config ini file
    config = ConfigParser.ConfigParser()
    config.read('logging_config.ini')
    trainFile = config.get('data','trainFile')
    testFile = config.get('data','testFile')

    # log init
    startLog(__name__)
    logger = logging.getLogger(__name__)
    #fld = trainFile.split('.')[0]
    #setLog(fld)
    
    # create the training & test sets, skipping the header row with [1:]
    logger.info('start') 

    # read csv file 

    # trainData = genfromtxt(open(trainFile,'r'), delimiter=',', dtype='f8')[1:] 
    # testData = genfromtxt(open(testFile,'r'), delimiter=',', dtype='f8')[1:]
    trainData = []
    with open(trainFile) as file:
        lines = csv.reader(file)
        for line in lines:
            trainData.append(line) # 42001 * 685
    trainData.remove(trainData[0])
    trainData = np.array(trainData)
    trainData = toFloat(trainData)

    testData = []
    with open(testFile) as file:
        lines = csv.reader(file)
        for line in lines:
            testData.append(line) # 42001 * 685
    testData.remove(testData[0])
    testData = np.array(testData)
    testData = toFloat(testData)
    # load done


    logger.info(['size of training data (#training_datapoints, #features/dimension + 1): ',
            np.shape(trainData)])
    # info:
    # shape (#training_datapoints, #features/dimension + 1), 
    # the first col is the output value of the training data
    # ie, each rows is inform: (output_i, input_i'(row vector))

    logger.info(['size of test data (N,1)', np.shape(testData)]) 
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
    logger.info('return y as array, X as array [N,D], trainData, testData') 
    return y,X,trainData,testData
	

def write_delimited_file(file_path, data,header=None, delimiter=","):
    f_out = open(file_path,"w")
    if header is not None:
        f_out.write(delimiter.join(header) + "\n")

    # get elements in data (a list)
    for line in data:
        if isinstance(line, str):
            f_out.write(line + "\n")
        elif isinstance(line, int):
            f_out.write(str(line) + "\n")
        else:
            f_out.write(delimiter.join(line) + "\n")
    f_out.close()


def writeCsv(result):
    """write submission file
        input should be list
    """ 
    # init
    config = ConfigParser.ConfigParser()
    config.read('logging_config.ini')
    saveFile = config.get('data','saveFile')
    startLog(__name__)
    logger = logging.getLogger(__name__)

    # init done

    write_delimited_file(saveFile, result)
    logger.info('save result as %s',saveFile)
    	





