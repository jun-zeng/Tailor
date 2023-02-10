import argparse
import logging
from colorlog import ColoredFormatter

logger = logging.getLogger(name=__name__)

def init_logger(level:int) -> None:
    formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(levelname)6s | %(log_color)s%(message)6s",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'yellow',
            'WARNING':  'green',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='driver',
                                     description='cpg constructor for c programs')
    
    parser.add_argument('-l', '--logging', type=int, default=20,
                        help='Log Level [10-50] (default: 10 - Debug)')
    parser.add_argument('--src_path', type=str, default='data/src_codes',
                        help='directory path of source codes')
    parser.add_argument('--encoding', default=False, action='store_true',
                        help='encode the cpgs or not')
    parser.add_argument('--encode_path', type=str, default='data/encode_res',
                        help='directory path of encode results')
    parser.add_argument('--iresult_path', type=str, default='data/inter_res',
                        help='directory path of inter results (function list & dict)')
    parser.add_argument('--store_iresult', default=False, action='store_true',
                        help='store inter results or not')
    parser.add_argument('--load_iresult', default=False, action='store_true',
                        help='load inter result or not')
    parser.add_argument('--statistics', default=False, action='store_true',
                        help='print cpg statistics or not')

    
    args = parser.parse_args()
    
    return args
def init_setting() -> argparse.Namespace:
    args = parse_args()
    init_logger(args.logging)
    
    return args