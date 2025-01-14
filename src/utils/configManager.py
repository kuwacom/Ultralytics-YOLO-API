import configparser
import logging

logger = logging.getLogger(__name__)

config = configparser.ConfigParser()

def loadConfig():
    try:
        logger.info('Read config.ini...')
        config.read('config.ini')
        return config
    except Exception as e:
        logger.error('config.ini not found')