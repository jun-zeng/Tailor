from utils.setting import logger
from utils.data_structure import Queue
from ccpg.cpg.ast_constructor import gen_ast_cpg
from ccpg.cpg.cfg_constructor import cfg_build
from ccpg.cpg.cg_node import CGNode, CalleeNode
from ccpg.cpg.ddg_constructor import ddg_build
from networkx import DiGraph



def check_edge(cpg: DiGraph, start: str, end: str) -> str:
    """Check the encoding of edge between start and end
    """
    if cpg.has_edge(start, end):
        return cpg[start][end]['edge_type']
    
    return False

def cg_dict_constructor(func_list: list) -> dict:
    """Parse all functions to construct func dict for call graph building.
    """
    if func_list == None:
        logger.error('CG_dict_constructor needs function list, exit')
        exit(-1)
    logger.info('Start generating function dict...')
    cg_dict = dict()
    logger.debug(len(func_list))
    for func in func_list:
        # logger.error(func.file_name + '-' + func.func_name)
        func_root = func.sast.root
        ast_cpg = gen_ast_cpg(func.sast)
        entrynode, fringe = cfg_build(ast_cpg, func_root)
        ddg_build(ast_cpg, func_root)
        callees = list_all_callees(ast_cpg, func_root, func.file_name)
        func_cgnode = CGNode(func.file_name, func.func_name, func.parameter_type, entrynode, fringe, ast_cpg, callees)
        key = func.file_name + '-' + func.func_name
        if key not in cg_dict.keys():
            cg_dict[key] = [func_cgnode]
        else:
            cg_dict[key].append(func_cgnode)
    logger.info('Finish generate function dict')
        
    return cg_dict

def list_all_callees(cpg: DiGraph, node: str, file_name: str) -> dict:
    """Given a code property graph instance and its root node, traverse all statements and find related callees.
    """
    callee_info = dict()

    visited = list()
    queue = Queue()
    queue.push(node)
    visited.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        callees = extract_callees(cpg, current_node, file_name)
        if current_node in callee_info:
            logger.error('Callees exist duplicate dict key, exit')
            exit(-1)
        if len(callees) > 0:
            callee_info[current_node] = callees
        
        node_successors = list(cpg.successors(current_node))
        for _successor in node_successors:
            if cpg.nodes[_successor]['cpg_node'].match_statement and _successor not in visited and check_edge(cpg, current_node, _successor) in ['010', '110', '011', '111']:
                queue.push(_successor)
                visited.append(_successor)
    
    return callee_info

def extract_callees(cpg: DiGraph, node: str, file_name: str) -> list:
    """Given a code property graph instance and one statement node, find callees rooted by this statement. 
    """
    callee_list = list()

    visited = list()
    queue = Queue()
    queue.push(node)
    visited.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        if cpg.nodes[current_node]['cpg_node'].node_type == 'call_expression':
            callee_node = extract_single_callee(cpg, node, current_node, file_name)
            callee_list.append(callee_node)
        
        node_successors = list(cpg.successors(current_node))
        for _successor in node_successors:
            if cpg.nodes[_successor]['cpg_node'].match_statement:
                continue
            if check_edge(cpg, current_node, _successor) in ['100'] and _successor not in visited:
                queue.push(_successor)
                visited.append(_successor)
    
    return callee_list

def extract_single_callee(cpg: DiGraph, stat_node: str, invocation_node: str, file_name: str) -> CalleeNode:
    """Extract callee information and construct calleenode.
    """
    node_successors = list(cpg.successors(invocation_node))

    # callee name 
    callee_name = cpg.nodes[node_successors[-2]]['cpg_node'].node_token

    # args list
    _callee_args = list(cpg.successors(node_successors[-1]))
    callee_args_num = len(_callee_args)

    callee_node = CalleeNode(stat_node, file_name, callee_name, callee_args_num)

    return callee_node
