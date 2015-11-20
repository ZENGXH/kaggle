"""useage:
	import my_log
	logger = my_log.setLog('name_of_file')
	# the name of the logger instance is 'log_global'
	# inside other function fitst get the logger by
	logger = logging.getLogger('log_global')
	logger.info('....')
"""
import logging

def setLog(logFile):
	print 'generate logfile: '+logFile+'.txt'
	logger = logging.getLogger('log_global')

	logger.setLevel(logging.DEBUG)
	logging.basicConfig(filename= logFile+'.txt',
                    level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	fh = logging.FileHandler(logFile+'.txt')
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	logger.debug('log setup done')
	return logger
