import os

from util.setting import log

def ensure_dir(dir_path:str) -> None:
    dir = os.path.dirname(dir_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def exist_dir(dir_path:str) -> None:
    dir = os.path.dirname(dir_path)
    if not os.path.isdir(dir):
        log.error("Dir does not exist: %s", dir)
        exit(-1)
