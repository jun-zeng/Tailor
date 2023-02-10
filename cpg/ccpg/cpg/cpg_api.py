from utils.setting import *
from ccpg.sast.src_parser import *
from ccpg.util.common import *
from ccpg.util.visualize import *
from ccpg.util.helper import *

from ccpg.cpg.ast_constructor import *
from ccpg.cpg.cfg_constructor import *
from ccpg.cpg.cg_constructor import *
from ccpg.cpg.ddg_constructor import *

def ast4singlefile(file_path: str) -> list:
    """Generate ast representations for functions in one file.
    """
    if file_path == None:
        logger.error('AST4singlefile lacks file path parameter.')
        exit(-1)
    file_funcs = c_parser(file_path)
    func_list = list()

    if has_entry_function(file_funcs):
        for func in file_funcs:
            if not func.has_type('ERROR'):
                ast_cpg = gen_ast_cpg(func.sast)
                func_list.append(ast_cpg)

    return func_list

def cpg4singlefile(file_path: str) -> list:
    """Generate cpg representation for functions in one file.
    """
    if file_path == None:
        logger.error('CPG4singlefile lacks file path parameter.')
        exit(-1)
    file_funcs = c_parser(file_path)
    func_list = list()

    if has_entry_function(file_funcs):
        for func in file_funcs:
            if not func.has_type('ERROR'):
                ast_cpg = gen_ast_cpg(func.sast)
                root = func.sast.root
                cfg_build(ast_cpg, root)
                ddg_build(ast_cpg, root)
                func_list.append(ast_cpg)
    
    return func_list

def extract_funcs(dir_path: str) -> list:
    """Extract all functions from multiple files
    """
    if dir_path == None:
        logger.error('Extract funcs lacks directory path.')
    
    files = traverse_src_files(dir_path, 'c')
    func_list = list()
    logger.info('Extract functions...')
    for file in files:
        logger.info(file)
        file_func = c_parser(file)
        if len(file_func) == 0:
            logger.warn(file)
            exit(-1)
        if not has_entry_function(file_func):
            logger.warn(file)
            exit(-1)
        for func in file_func:
            if not func.has_type('ERROR'):
                func_list.append(func)
            else:
                logger.error(file)
                logger.error(func.func_name)
                exit(-1)
    
    return func_list

def cpg4multifiles(func_list: list) -> dict:
    """Generate cpg for multi files.
    """
    logger.info('Generate CPG Dict...')
    cpg_dict = cg_dict_constructor(func_list)

    return cpg_dict