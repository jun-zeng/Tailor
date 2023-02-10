from utils.data_structure import Stack
from utils.setting import logger
from javacpg.sast.ast_builder import build_func_sast
from javacpg.sast.ast_parser import ASTParser
from javacpg.sast.fun_unit import FunUnit
from javacpg.sast.query_pattern import JAVA_QUERY

exclude_type = [",","{",";","}",")","(",'"',"'","`",""," ","[]","[","]",":",".","''","'.'","b", "\\", "'['", "']","''", "comment", "@", "?"]

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
        logger.debug('Can not extract file name for path: {}' .format(file_path))
        exit(-1)
    
    return file_name

def align_query_result(query_result: list, meta1: str, meta2: str) -> list:
    """Split fused query results and align them in different lists

    attributes:
        query_result -- a list of query results.
        meta1 -- label of the first category.
        meta2 -- label of the second category.
    
    returns:
        align_results -- [_meta1, _meta2], a list contains two list, and first is for meta1, second is for meta2
    """
    length = len(query_result)
    _meta1 = []
    _meta2 = []
    meta1_stack = Stack()
    for idx in range(length):
        if query_result[idx][1] == meta1:
            meta1_stack.push(query_result[idx][0])
        elif query_result[idx][1] == meta2:
            if meta1_stack.is_empty():
                logger.error('Align empty stack, exit.')
                exit(-1)
            _meta1.append(meta1_stack.pop())
            _meta2.append(query_result[idx][0])
    
    if len(_meta1) != len(_meta2):
        logger.debug('Meta1 and Meta2 query results are not matched. Exit')
        exit(-1)

    return [_meta1, _meta2]

def java_parser(file_path: str) -> list:
    """ Parse Java source code file & extract function unit

    attributes:
        file_path -- the path of Java source file.
    
    returns:
        func_list -- list including all function in current file.
    """
    func_list = []
    parser = ASTParser('java')
    with open(file_path, 'rb') as f:
        serial_code = f.read()
        code_ast = parser.parse(serial_code)
    
    root_node = code_ast.root_node

    # print(root_node.sexp())

    # obtain file name
    file_name = extract_filename(file_path)

    query = JAVA_QUERY()
    # query import headers (e.g., import java.util.Scanner)
    import_header = query.import_header_query().captures(root_node)
    import_header = [serial_code[x[0].start_byte:x[0].end_byte-1].decode('utf8') for x in import_header]

    # query field parameters (e.g., class variables)
    field_params = query.class_filed_query().captures(root_node)
    field_params = [serial_code[x[0].start_byte:x[0].end_byte].decode('utf8') for x in field_params]

    # query methods
    _methods = query.class_method_query().captures(root_node)
    
    for _method in _methods:
        _m_name_tmp = query.method_declaration_query().captures(_method[0])
        _m_name = serial_code[_m_name_tmp[0][0].start_byte:_m_name_tmp[0][0].end_byte].decode('utf8')
        _m_param_tmp = query.method_parameters_query().captures(_method[0])
        _m_param_type, _m_param_name = align_query_result(_m_param_tmp, 'type', 'name')
        _m_param_type = [serial_code[x.start_byte:x.end_byte].decode('utf8') for x in _m_param_type]
        _m_param_name = [serial_code[x.start_byte:x.end_byte].decode('utf8') for x in _m_param_name]
        sast = build_func_sast(file_name, _m_name, _method[0], serial_code, exclude_type)

        cur_func = FunUnit(sast, file_name, _m_name, _m_param_type, _m_param_name, import_header, field_params)
        func_list.append(cur_func)
    
    return func_list

