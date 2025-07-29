import logging
from logging.handlers import RotatingFileHandler

myLog = None

def getLog():
    global myLog
    if not myLog:
        log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s (%(filename)s,%(funcName)s,%(lineno)d) ', '%H:%M:%S')
        
        logFile = 'main.log'
        
        my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=1*1024*1024, backupCount=2, encoding=None, delay=0)
        my_handler.setFormatter(log_formatter)
        my_handler.setLevel(logging.INFO)
        
        app_log = logging.getLogger('root')
        app_log.setLevel(logging.INFO)
        
        app_log.addHandler(my_handler)
        
        myLog = app_log
    
    return myLog
    
