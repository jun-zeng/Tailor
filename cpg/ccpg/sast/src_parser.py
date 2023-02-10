from ccpg.sast.ast_builder import build_func_sast
from utils.setting import logger
from ccpg.sast.ast_parser import ASTParser
from ccpg.sast.query_pattern import C_QUERY
from ccpg.sast.fun_unit import FunUnit
import os

exclude_type = [",","{",";","}",")","(",'"',"'","`",""," ","[]","[","]",":",".","''","'.'","b", "\\", "'['", "']","''", "comment", "->", "escape_sequence", "@", "?", ""]

def extract_filename(file_path: str) -> str:
    """Extract file name excluding extension from file path.

    attributes:
        file_path -- the path of current file.
    
    returns:
        file_name -- the name of current file.
    """
    file_name = ''
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]

    if file_name == '':
        logger.error('Can not extract file name for path: {}' .format(file_path))
        exit(-1)
    
    logger.debug('Parsing file: {}' .format(file_name))
    return file_name

def c_parser(file_path: str) -> list:
    """Parse C source code file and extract function unit
    
    attributes:
        file_path -- the path of C source file.
    
    returns:
        func_list -- list including all functions in one file.
    """
    func_list = list()
    
    # obtain file name
    file_name = extract_filename(file_path)
    
    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    lib_path = path + '/lib/my-languages.so'
    parser = ASTParser(lib_path)
    
    with open(file_path, 'rb') as f:
        serial_code = f.read()
        code_ast = parser.parse(serial_code)
    
    root_node = code_ast.root_node
    # print(code_ast.root_node.sexp())
    
    query = C_QUERY()
    
    # query include paths (e.g., include <stdlib.h>)
    include_path = query.include_path_query().captures(root_node)
    include_path = [serial_code[x[0].start_byte+1:x[0].end_byte-1].decode('utf8') for x in include_path]
    logger.debug('Include paths: ({})'.format(', '.join(include_path)))
    
    # query functions
    functions = query.function_definition_query().captures(root_node)
    
    length = len(functions)
    for idx in range(length):
        _func_name = query.function_declarator_query().captures(functions[idx][0])
        _func_name = serial_code[_func_name[0][0].start_byte:_func_name[0][0].end_byte].decode('utf8')
        logger.debug('Parsing function ({}) in file ({})' .format(_func_name, file_name))
        
        _func_type = query.function_parameters_type_query().captures(functions[idx][0])
        _func_type = [serial_code[x[0].start_byte:x[0].end_byte].decode('utf-8') for x in _func_type]
        
        logger.debug('Parameters of function ({}): ({})' .format(_func_name, ', '.join(_func_type)))
        
        sast = build_func_sast(file_name, _func_name, functions[idx][0], serial_code, exclude_type)
        
        cur_func = FunUnit(sast, file_name, _func_name, _func_type, include_path)
        
        func_list.append(cur_func)
    
    return func_list