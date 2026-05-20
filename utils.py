import os
import logging
import math
from config import *
def get_logger(logs_dir = 'logs'):
  os.makedirs(logs_dir, exist_ok = True)
  logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
      logging.FileHandler(os.path.join(logs_dir, 'training.log')),
      logging.StreamHandler()
    ]
  )
  return logging.getLogger()
