"""This file provides some simple apis of code property graph.
Note: For Java Language, the input must be class file.
"""
from javacpg.cpg.ast_constructor import *
from javacpg.cpg.cfg_constructor import *
from javacpg.cpg.cg_constructor import *
from javacpg.cpg.ddg_constructor import *
from javacpg.encoding.encoding import *
from javacpg.encoding.query import *
from javacpg.sast.src_parser import *
from javacpg.util.common import *
from javacpg.util.helper import *
from javacpg.util.visualize import *
from utils.setting import logger
from func_timeout import func_set_timeout, FunctionTimedOut

@func_set_timeout(80)
def ast4singleclass(file_path: str = None) -> list:
    """Generate ast representations for functions in one class.
    """
    if file_path == None:
        logger.error('AST4singleclass lacks file path parameter.')
        exit(-1)
    file_funcs = java_parser(file_path)
    func_list = list()

    for func in file_funcs:
        if not func.has_type('ERROR'):
            ast_cpg = gen_ast_cpg(func.sast)
            func_list.append(ast_cpg)

    return func_list

@func_set_timeout(100)
def cpg4singleclass(file_path: str = None) -> list:
    """Generate cpg representations for functions in one class.
    """
    if file_path == None:
        logger.error('CPG4singleclass lacks fle path parameter.')
        exit(-1)
    
    file_funcs = java_parser(file_path)
    func_list = list()
    for func in file_funcs:
        logger.info(func.file_name, func.func_name)
        if not func.has_type('ERROR'):
            ast_cpg = gen_ast_cpg(func.sast)
            root = func.sast.root
            cfg_build(ast_cpg, root)
            ddg_build(ast_cpg, root)
            func_list.append(ast_cpg)
    logger.info(len(file_funcs))
    return func_list

@func_set_timeout(60)
def cpg_constructor(func: FunUnit) -> DiGraph:
    """Construct cpg.
    """
    func_root = func.sast.root
    ast_cpg = gen_ast_cpg(func.sast)
    cfg_build(ast_cpg, func_root)
    ddg_build(ast_cpg, func_root)

    return ast_cpg

"""
def extract_funcs(dir_path: str = None) -> list:
    if dir_path == None:
        logger.error('Cpg4multiclass lacks directory path.')
        exit(-1)
    files = traverse_src_files(dir_path, 'java')
    func_list = list()
    logger.info('Extract functions...')

    for file in files:
        file_func = java_parser(file)
        logger.info(f'Parsing file: {file}')
        if len(file_func) == 0:
            logger.warn(f'Cannot extract functions from {file}')
            exit(-1)
        for func in file_func:
            if not func.has_type('ERROR'):
                func_list.append(func)
                try:
                    cpg_constructor(func)
                    # func_list.append(func)
                    # print(file)
                except FunctionTimedOut:
                    logger.warn('Parsing file: {} \t function: {}' .format(file, func.func_name))

                
            else:
                logger.error(file)
                logger.error('Parsing file: {} \t function: {}' .format(file, func.func_name))

    return func_list
"""

def extract_funcs(dir_path: str = None) -> list:
    """Extract all functions from multiple classes (directory).
    """
    if dir_path == None:
        logger.error('Cpg4multiclass lacks directory path.')
        exit(-1)
    files = traverse_src_files(dir_path, 'java')
    func_list = list()
    logger.info('Extract functions...')

    for file in files:
        file_func = java_parser(file)
        logger.info(f'Parsing file: {file}')
        if len(file_func) == 0:
            logger.warn(f'Cannot extract functions from {file}')
            exit(-1)
        for func in file_func:
            if not func.has_type('ERROR'):
                func_list.append(func)
            else:
                logger.error(f'File: {file} \t function: {func.func_name} has ERROR Type.')
                exit(-1)

    return func_list

def cpg4tell(func_list: list = None) -> dict:
    """Generate cpg representations for functions in multiple classes. 

    attributes:
        func_list -- list of functions.
    
    returns:
        cpg_dict -- keys are function name + arguments number, values are CGNode instances. (please refer to the data structure of CGNode)
    """
    
    print('Generate CPG Dict...')
    cpg_dict = cg_dict_constructor(func_list)

    return cpg_dict

def cpg4clone(func_list: list = None) -> None:
    """Convert functions in func_list to cpg for code clone task.
    """
    if func_list == None:
        logger.error('No functions for code clone task.')
        exit(-1)
    
    cpg_list = list()

    for func in func_list:
        logger.info('Parsing file: {} \t function: {}' .format(func.file_name, func.func_name))

        func_root = func.sast.root
        ast_cpg = gen_ast_cpg(func.sast)
        entrynode, fringe = cfg_build(ast_cpg, func_root)
        ddg_build(ast_cpg, func_root)
        cpg_list.append(ast_cpg)
    
    logger.warn(f'cpg number: {len(cpg_list)}')

def cpg4multifiles(func_list: list) -> dict:
    """Generate cpg for multi files.
    """
    logger.info('Start generating CPG Dict...')
    cpg_dict = cg_dict_constructor(func_list)

    return cpg_dict