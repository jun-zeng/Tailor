from networkx import DiGraph
from utils.data_structure import Queue
from utils.setting import logger
from javacpg.cpg.ast_constructor import gen_ast_cpg
from javacpg.cpg.cfg_constructor import cfg_build
from javacpg.cpg.cg_node import CalleeNode, CGNode
from javacpg.cpg.ddg_constructor import ddg_build, merge_def_use

int_dict = ['decimal_integer_literal', 'hex_integer_literal', 'octal_integer_literal', 'binary_integer_literal']
float_dict = ['decimal_floating_point_literal', 'hex_floating_point_literal']

def check_edge(cpg: DiGraph = None, start: str = None, end: str = None) -> str:
    """Check the encoding of edge between start and end.
    """
    if cpg.has_edge(start, end):
        return cpg[start][end]['edge_type']
    
    return False

def cg_dict_constructor(func_list: list = None) -> dict:
    """Parse all functions extracted from files and store them in dict for later use.

    attributes:
        func_list -- list of functions, function is the instance of FunUnit.
    
    returns:
        func_dict -- dict of functions, key is function name and value is a list including different CGNodes.
    
    Note: Currently, we use function name + args number as dict key.
    """
    if func_list == None:
        logger.error('CG_dict_constructor needs function list, exit.')
        exit(-1)
    
    cg_dict = dict()

    for func in func_list:
        # construct CGNode
        func_root = func.sast.root
        ast_cpg = gen_ast_cpg(func.sast)
        entrynode, fringe = cfg_build(ast_cpg, func_root)
        ddg_build(ast_cpg, func_root)
        callees = list_all_callees(ast_cpg, func_root)
        func_cgnode = CGNode(func.file_name, func.import_header, func.func_name, func.parameter_type, func.parameter_name, entrynode, fringe, ast_cpg, callees)

        key = func.func_name + '-' + str(len(func.parameter_type))
        if key not in cg_dict.keys():
            cg_dict[key] = [func_cgnode]
        else:
            cg_dict[key].append(func_cgnode)

    logger.info('Finish generating function dict')


    return cg_dict


def list_all_callees(cpg: DiGraph = None, node: str = None) -> dict:
    """Given a cfg-based code property graph instance and its root node, traverse all statements and find related callees.

    attributes:
        cpg -- cfg-based code property graph for one function.

    returns:
        callee_dict -- dict that keys are callee function names, values are CalleeNodes.
    """
    callee_info = dict()

    visited = list()
    queue = Queue()
    queue.push(node)
    visited.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        callees = extract_callees(cpg, current_node)
        if current_node in callee_info:
            logger.error('Callees exist duplicate dict key, exit.')
            exit(-1)
        if len(callees) > 0:
            callee_info[current_node] = callees
        
        node_successors = list(cpg.successors(current_node))
        for _successor in node_successors:
            if cpg.nodes[_successor]['cpg_node'].match_statement and _successor not in visited and check_edge(cpg, current_node, _successor) in ['010', '110', '011', '111']:
                queue.push(_successor)
                visited.append(_successor)
    
    return callee_info


def extract_callees(cpg: DiGraph = None, node: str = None) -> list:
    """Given a code property graph instance and one statement node, find callees rooted by this statement.
    """
    callee_list = list()

    visited = list()
    queue = Queue()
    queue.push(node)
    visited.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        if cpg.nodes[current_node]['cpg_node'].node_type == 'method_invocation':
            callee_node = extract_single_callee(cpg, node, current_node)
            callee_list.append(callee_node)
        node_successors = list(cpg.successors(current_node))
        
        for _successor in node_successors:
            if cpg.nodes[_successor]['cpg_node'].match_statement:
                continue
            if check_edge(cpg, current_node, _successor) in ['100'] and _successor not in visited:
                queue.push(_successor)
                visited.append(_successor)
    
    return callee_list

def extract_single_callee(cpg: DiGraph, stat_node: str, invocation_node: str):
    """Given one method invocation node, extract basic information.
    """
    """
    method_invocation: $ => seq(
      choice(
        field('name', choice($.identifier, $._reserved_identifier)),
        seq(
          field('object', choice($.primary_expression, $.super)),
          '.',
          optional(seq(
            $.super,
            '.'
          )),
          field('type_arguments', optional($.type_arguments)),
          field('name', choice($.identifier, $._reserved_identifier)),
        )
      ),
      field('arguments', $.argument_list)
    ),

    argument_list: $ => seq('(', commaSep($.expression), ')'),
    """

    node_successors = list(cpg.successors(invocation_node))

    # callee name
    _callee_node = node_successors[-2]
    call_name = cpg.nodes[_callee_node]['cpg_node'].node_token

    # args_list
    _callee_args = list(cpg.successors(node_successors[-1]))
    callee_args_num = len(_callee_args)
    callee_args_type = list()
    # keep simple type variables of current callee
    for _args in _callee_args:
        _args_type = cpg.nodes[_args]['cpg_node'].node_type
        if _args_type in int_dict:
            callee_args_type.append('int')
        elif _args_type in float_dict:
            callee_args_type.append('float')
        elif _args_type == 'character_literal':
            callee_args_type.append('Char')
        elif _args_type == 'string_literal':
            callee_args_type.append('String')
        elif _args_type in ['true', 'false']:
            callee_args_type.append(_args_type)
        elif _args_type == 'identifier':
            _args_var = cpg.nodes[_args]['cpg_node'].node_token
            _recover_type = type_recovery(cpg, _args_var, stat_node)
            if _recover_type:
                callee_args_type.append(_recover_type)
        else:
            continue
    
    callee_node = CalleeNode(stat_node, call_name, callee_args_num, callee_args_type)
    
    return callee_node

def type_recovery(cpg: DiGraph, args_var: str, stat_node: str) -> str:
    """Given one callee args and do simple type recovery.
    """
    # back tracking according to data flow edge.
    recovery_type = None

    visited = list()
    queue = Queue()
    queue.push(stat_node)
    visited.append(stat_node)

    while not queue.is_empty():
        current_node = queue.pop()
        node_predecessors = list(cpg.predecessors(current_node))
        for _predecessor in node_predecessors:
            if cpg.nodes[_predecessor]['cpg_node'].match_statement and _predecessor not in visited and check_edge(cpg, _predecessor, current_node) in ['001', '011', '101', '111']:
                queue.push(_predecessor)
                visited.append(_predecessor)
        
        # we filter assignment expression, only focus on declaraion statement.
        if cpg.nodes[current_node]['cpg_node'].node_type == 'assignment_expression':
            continue
        ddg_node = merge_def_use(cpg, current_node)
        if args_var in ddg_node.defs:
            recovery_type = identify_type(cpg, current_node, args_var)
            break
    
    return recovery_type

def identify_type(cpg: DiGraph, node: str, args_var: str) -> str:
    """Extract type of args variable.
    """

    visited = list()
    queue = Queue()
    queue.push(node)
    visited.append(node)

    var_node = None

    while not queue.is_empty():
        current_node = queue.pop()
        current_node_token = cpg.nodes[current_node]['cpg_node'].node_token
        if current_node_token == args_var:
            var_node = current_node
            break
        node_successors = cpg.successors(current_node)
        for _successor in node_successors:
            if not cpg.nodes[_successor]['cpg_node'].match_statement and _successor not in visited:
                queue.push(_successor)
                visited.append(_successor)
    
    if var_node is None:
        logger.error('Cannot find variable declaration, exit')
        exit(-1)
    
    identified_type = None
    # back tracking to find type
    back_queue = Queue()
    back_visited = list()
    back_queue.push(var_node)
    back_visited.append(var_node)

    while not back_queue.is_empty():
        current_node = back_queue.pop()
        current_node_type = cpg.nodes[current_node]['cpg_node'].node_type
        if current_node_type == 'local_variable_declaration':
            cur_node_successors = list(cpg.successors(current_node))
            cur_node_successors.reverse()
            for _cur_successor in cur_node_successors:
                _cur_successor_type = cpg.nodes[_cur_successor]['cpg_node'].node_type
                if _cur_successor_type != 'variable_declarator' and check_edge(cpg, current_node, _cur_successor) in ['100']:
                    if _cur_successor_type == 'type_identifier':
                        identified_type = cpg.nodes[_cur_successor]['cpg_node'].node_token
                    elif _cur_successor_type in ['integral_type', 'floating_point_type', 'boolean_type', 'void_type']:
                        _wrapper_successors = list(cpg.successors(_cur_successor))
                        if len(_wrapper_successors) == 0:
                            identified_type = cpg.nodes[_cur_successor]['cpg_node'].node_token
                        else:
                            identified_type = cpg.nodes[_wrapper_successors[0]]['cpg_node'].node_type
                    else:
                        identified_type = _cur_successor_type
                    break
            break
        elif current_node_type == 'formal_parameter':
            cur_node_successors = list(cpg.successors(current_node))
            wrapper_type = cpg.nodes[cur_node_successors[-2]]['cpg_node'].node_type
            if wrapper_type in ['integral_type', 'floating_point_type', 'boolean_type', 'void_type']:
                _wrapper_successors = list(cpg.successors(cur_node_successors[-2]))
                if len(_wrapper_successors) == 0:
                    identified_type = cpg.nodes[cur_node_successors[-2]]['cpg_node'].node_token
                else:
                    identified_type = cpg.nodes[_wrapper_successors[0]]['cpg_node'].node_type
            else:
                identified_type = wrapper_type
            break
        else:
            # now we only support local variable declaraion and formal parameter.
            pass
        cur_preds = list(cpg.predecessors(current_node))
        for _pred in cur_preds:
            if _pred not in back_visited and check_edge(cpg, _pred, current_node) in ['100']:
                back_queue.push(_pred)
                back_visited.append(_pred)

    return identified_type
