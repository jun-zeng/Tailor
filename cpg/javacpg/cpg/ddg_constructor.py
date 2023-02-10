from networkx import DiGraph
from utils.data_structure import Queue, Stack
from utils.setting import logger
from javacpg.sast.fun_unit import FunUnit
from javacpg.util.common import cpg_edge_type
from javacpg.cpg.ddg_node import DDGNode

def ddg_build(cfg_cpg: DiGraph, node: str = None):
    """Identify def-use information in control-flow based Code Property Graph and generate data flow edge for cfg_cpg to form complete Code Property Graph.

    attribtues:
        cfg_cpg -- an instance of function.
        node -- root node of the cfg_cpg
    
    returns:
        xxxx
    """
    def_use_chain = gen_def_use_chain(cfg_cpg, node)
    for key, value in def_use_chain.items():
        # print(value.print_defs_uses())
        if value.uses == []:
            continue
        for use_var in value.uses:
            back_tracking(cfg_cpg, key, def_use_chain, use_var)
        

def has_dd_relation(use_node: DDGNode = None, def_node: DDGNode = None) -> bool:
    """Determine whether two ddg nodes have data flow dependency. In general, we consider whether use nodes in use_node have been defined in def_node.

    attributes:
        use_node -- one DDGNode instance
        def_node -- one DDGNode instance
    
    returns:
        True / False -- if these two nodes have relationship return true, else return false.
    """
    if use_node == None or def_node == None:
        logger.error('Use Node or Def Node is None, Exit')
        exit(-1)
    
    uses = use_node.get_uses()
    defs = def_node.get_defs()

    _dd_rel = False
    for _use in uses:
        if _use in defs:
            _dd_rel = True
            break
    
    return _dd_rel

def has_dd_rel(use_var: str = None, def_node: DDGNode = None) -> bool:
    """Check whether use_var is defined by def_node.
    """

    if use_var == None or def_node == None:
        logger.error('Use variable or Def Node is None, Exit.')
        exit(-1)
    defs = def_node.get_defs()
    _dd_rel = False
    if use_var in defs:
        _dd_rel = True
    return _dd_rel

def add_ddg_edge(cfg_cpg: DiGraph = None, start_node_identifier: str = None, end_node_identifier: str = None) -> bool:
    """Add data flow edge into cfg_cpg to form complete Code Property Graph.

    attributes:
        cfg_cpg -- an instance cfg-based code property graph.
        start_node -- node identifier
        end_node -- node identifier
    
    returns:
        True / False -- if add the edge successfully, return True, else False.
    """
    edge_type = cpg_edge_type(cfg_cpg, start_node_identifier, end_node_identifier, '001')

    cfg_cpg.add_edge(start_node_identifier, end_node_identifier, edge_type = edge_type)

    return True

def initialize_def(func: FunUnit = None) -> list:
    """Initialize def information with class field parameters.

    attributes:
        func -- an instance of FunUnit.
    
    returns:
        def_uses -- a list contains class field parameters.
    """
    return func.field_params
    
def check_edge(cfg_cpg: DiGraph = None, start: str = None, end: str = None) -> str:
    """Check the encoding of edge between start and end.
    """
    if cfg_cpg.has_edge(start, end):
        return cfg_cpg[start][end]['edge_type']
    
    return False


def gen_def_use_chain(cfg_cpg: DiGraph = None, node: str = None) -> dict:
    """Traverse cfg cpg and extract def-use information for each statement node in order. Remember we traverse the graph by level-order.

    attributes:
        cfg_cpg -- cfg-based code property graph for one function.

    returns:
        [DDGNode] -- list of DDGNode, each DDGNode represents one statement's def-use information.
    """

    ddg_chain = dict()

    # used to check whether visit current node or not to avoid circle.
    check_exist = list()
    queue = Queue()
    queue.push(node)
    check_exist.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        ddg_node = merge_def_use(cfg_cpg, current_node)
        if ddg_node.node_key in ddg_chain:
            logger.error('Appear duplicated Dict key, exit.')
            exit(-1)
        ddg_chain[ddg_node.node_key] = ddg_node
        node_successors = list(cfg_cpg.successors(current_node))
        for _successor in node_successors:
            if cfg_cpg.nodes[_successor]['cpg_node'].match_statement and _successor not in check_exist and check_edge(cfg_cpg, current_node, _successor) in ['010', '110', '011', '111']:
                queue.push(_successor)
                check_exist.append(_successor)

    return ddg_chain

def back_tracking(cfg_cpg: DiGraph = None, node: str = None, def_use_chain: dict = None, use_var: str = None) -> bool:
    """Given cfg-based Code Property Graph and a node identifier, then do back tracking to add data flow edge.
    """
    visited = list()
    queue = Queue()
    if node == None:
        return True

    queue.push(node)
    while not queue.is_empty():
        current_node = queue.pop()
        visited.append(current_node)
        predecessors = list(cfg_cpg.predecessors(current_node))
        for _predecessor in predecessors:
            _match_stat = cfg_cpg.nodes[_predecessor]['cpg_node'].match_statement
            if _match_stat and _predecessor not in visited and check_edge(cfg_cpg, _predecessor, current_node) in ['010', '110', '011', '111']:
                if _predecessor not in def_use_chain:
                    logger.error('Current node cannot be find in def use chain, exit')
                    exit(-1)
                ddg_predecessor = def_use_chain[_predecessor]
                if has_dd_rel(use_var, ddg_predecessor):
                    add_ddg_edge(cfg_cpg, ddg_predecessor.node_key, node)
                else:
                    queue.push(_predecessor)
    
    # print back tracking path
    # for x in visited:
    #     print(cfg_cpg.nodes[x]['cpg_node'].node_type, end=' ')
    # print()
    return True

def gen_def_use_chains(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Traverse cfg-based cpg and extract def-use information for each statement node in order. Note we traverse the graph with customized DFS to obtain all possible path.

    Note: this function can generate all def-use chains, since it simulates all execution paths. However, the time overhead is too large for us, hence, we construst the def use chain via other approach.

    attribtues:
        cfg_cpg -- cfg-based code property graph for one function.
        node -- root node of the cfg-based code property graph, also the source node
    
    returns:
        [[DDGNodes]] -- list of DDGNode list, each DDGNode list represents one def-use chain, each DDGNode represents one statement's def-use information.
    """

    def is_visited(visited: list, node: str) -> bool:
        res = False
        for _item in visited:
            if node == _item[0]:
                res = True
                break
        return res

    def is_leaf(cfg_cpg: DiGraph = None, visited: list = None, current_node: str = None) -> bool:
        res = True

        _successors = list(cfg_cpg.successors(current_node))
        if len(_successors) == 0:
            res = True
        else:
            for _successor in _successors:
                if not is_visited(visited, _successor):
                    res = False
                    break
        
        return res
        

    ddg_chains = list()

    # used to check whether the current node are visited or not to avoid circle.
    visited = list()

    stack = Stack()
    
    # used to label the level to ensure we can traverse all paths. 
    level = 0
    node_info = [node, level]
    stack.push(node_info)
    

    while not stack.is_empty():
        current_node = stack.pop()

        wait_del = []
        for _vis in visited:
            if _vis[1] >= current_node[1]:
                wait_del.append(_vis)
        for _v in wait_del:
            visited.remove(_v)
        
        # complete one path
        if is_leaf(cfg_cpg, visited, current_node[0]):
            ddg_chains.append(visited + [current_node])

        node_successors = list(cfg_cpg.successors(current_node[0]))
        node_successors.reverse()

        for _successor in node_successors:
            if not cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
                continue
            if is_visited(visited, _successor):
                continue
            if not check_edge(cfg_cpg, current_node[0], _successor) in ['010', '110']:
                continue

            _successor_info = [_successor, current_node[1] + 1]
            stack.push(_successor_info)
        visited.append(current_node)
        
    return ddg_chains


def combine_fields(left_v: list = None, right_v: list = None) -> list:
    """Combine two variable list and generate new def use variable.
    """
    res_v = list()

    if left_v == [] or res_v == []:
        return res_v

    longest_pre = left_v[0]
    for l_v in left_v:
        if len(l_v) > len(longest_pre):
            longest_pre = l_v
    for r_v in right_v:
        res_v.append(longest_pre + '.' + r_v)

    return res_v

def merge_def_use(cfg_cpg: DiGraph = None, node: str = None) -> DDGNode:
    """Parse one statement node, and traverse its all child entities (non cfg) to collect its def use information and Merge them together.
    """
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        defs += _s_def
        uses += _s_use
        unknown += _s_unknown

    tmp_defs = list(set(defs))
    tmp_uses = list(set(uses))
    tmp_unknown = list(set(unknown))

    tmp_defs.sort(key=defs.index)
    tmp_uses.sort(key=uses.index)
    tmp_unknown.sort(key=unknown.index)

    # further parse return statement, since cannot handle return in general logic.
    node_type = cfg_cpg.nodes[node]['cpg_node'].node_type
    if node_type == 'return_statement':
        tmp_uses = tmp_uses + tmp_unknown
        tmp_unknown = []
    elif node_type == 'enhanced_for_statement':
        tmp_defs = tmp_defs + tmp_unknown
        tmp_unknown = []
    
    return DDGNode(node, cfg_cpg.nodes[node]['cpg_node'].node_type, tmp_defs, tmp_uses, tmp_unknown)

def extract_def_use(cfg_cpg : DiGraph = None, node: str = None) -> list:
    """Traverse entity of statement identified by node to extract its def-use information.

    attributes:
        cfg_cpg -- cfg-based code property graph for one function.
        node -- identifier of CPGNode belonging to statement.
    
    returns:
        ddg_node -- a DDGNode instance including node and def-use information.
    """
    node_type = cfg_cpg.nodes[node]['cpg_node'].node_type
    if node_type == 'binary_expression':
        defs, uses, unknown = ddg_binary_expression(cfg_cpg, node)
    elif node_type == 'update_expression':
        defs, uses, unknown = ddg_update_expression(cfg_cpg, node)
    elif node_type == 'assignment_expression':
        defs, uses, unknown = ddg_assignment_expression(cfg_cpg, node)
    elif node_type == 'variable_declarator':
        defs, uses, unknown = ddg_variable_declarator(cfg_cpg, node)
    elif node_type == 'object_creation_expression':
        defs, uses, unknown = ddg_object_creation_expression(cfg_cpg, node)
    elif node_type == 'field_access':
        defs, uses, unknown = ddg_field_access(cfg_cpg, node)
    elif node_type == 'array_access':
        defs, uses, unknown = ddg_array_access(cfg_cpg, node)
    elif node_type == 'argument_list':
        defs, uses, unknown = ddg_argument_list(cfg_cpg, node)
    elif node_type == 'method_invocation':
        defs, uses, unknown = ddg_method_invocation(cfg_cpg, node)
    elif node_type == 'formal_parameter':
        defs, uses, unknown = ddg_formal_parameter(cfg_cpg, node)
    elif node_type == 'identifier':
        defs, uses, unknown = ddg_identifier(cfg_cpg, node)
    elif node_type == 'instanceof_expression':
        defs, uses, unknown = ddg_instanceof_expression(cfg_cpg, node)
    elif node_type == 'ternary_expression':
        defs, uses, unknown = ddg_ternary_expression(cfg_cpg, node)
    elif node_type == 'unary_expression':
        defs, uses, unknown = ddg_unary_expression(cfg_cpg, node)
    elif node_type == 'array_creation_expression':
        defs, uses, unknown = ddg_array_creation_expression(cfg_cpg, node)
    elif node_type == 'cast_expression':
        defs, uses, unknown = ddg_cast_expression(cfg_cpg, node)
    elif node_type == 'parenthesized_expression':
        defs, uses, unknown = ddg_parenthesized_expression(cfg_cpg, node)
    else:
        defs, uses, unknown = ddg_deeper(cfg_cpg, node)
    
    return [defs, uses, unknown]

def ddg_deeper(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse node not be expression.
    """
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    
    if len(node_successors) == 0:
        logger.debug('Current node has no children, just return null')
        return [defs, uses, unknown]
    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        defs += _s_def
        uses += _s_use
        unknown += _s_unknown
    
    return [defs, uses, unknown]

"""
handle primary expression
"""
# identifier
def ddg_identifier(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse identifier and extract def use information. Specifically, identifier cannot reflect its def use information.
    """
    defs, uses = [], []
    unknown = [cfg_cpg.nodes[node]['cpg_node'].node_token]
    
    return [defs, uses, unknown]

# _reversed_identifier
# TODO

# object creation expression
def ddg_object_creation_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse object creation expression and extract def use information. Specifically, object creation expression only has used variables.
    """
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _s_def + _s_use + _s_unknown
    
    return [defs, uses, unknown]

# filed_access
def ddg_field_access(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse field access and extract def use information. Specifically, field access only has used variables. Note: for field access need combinations(e.g., a.b.c, use a, a.b a.b.c)
    """
    defs, uses, unknown = [], [], []
    field_access_successors = list(cfg_cpg.successors(node))
    # In general, field access node has two children.
    if not len(field_access_successors) == 2:
        logger.error('Field access has {} children, exit' .format(len(field_access_successors)))
        exit(-1)
    
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, field_access_successors[0])
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, field_access_successors[-1])
    
    defs = _r_def + _r_use + _r_unknown
    uses = _l_def + _l_use + _l_unknown
    combinations = combine_fields(uses, defs)
    defs = combinations
    unknown = []

    return [defs, uses, unknown]

# array_access
def ddg_array_access(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse array access and extract def use information. Specifically, array access only has used variables.
    """
    defs, uses, unknown = [], [], []

    array_access_successors = list(cfg_cpg.successors(node))
    # array_access has two children
    if len(array_access_successors) != 2:
        logger.error('Array access has {} children.' .format(len(array_access_successors)))
        exit(-1)
    
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, array_access_successors[0])
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, array_access_successors[-1])

    uses = _l_def + _l_use + _l_unknown + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# method_invocation
def ddg_method_invocation(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse method invocation and extract def use information. Specifically, method invocation only has used variables.
    """
    method_invocation_successors = list(cfg_cpg.successors(node))

    defs, uses, unknown = [], [], []
    
    for _successor in method_invocation_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _s_def + _s_use + _s_unknown
    
    return [defs, uses, unknown]

# method_reference
# TODO

# array_creation_expression
def ddg_array_creation_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse array_creation_expression and extract def use information. Specifically, array creation expression only has use variables.
    """
    defs, uses, unknown = [], [], []
    array_creation_successors = list(cfg_cpg.successors(node))

    for _successor in array_creation_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _s_def + _s_use + _s_unknown
    
    return [defs, uses, unknown]

"""
handle expression
"""
# assignment_expression
def ddg_assignment_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse assignment expression and extract def use information.
    Specifically, assignment expression has both def and use variables.
    """
    defs, uses, unknown = [], [], []

    # assignment expression has three children
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        logger.error('Assignment expression has {} children.' .format(len(node_successors)))
        exit(-1)
    _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
    defs = defs + _f_def + _f_unknown 
    uses = uses + _f_use

    _t_def, _t_use, _t_unknown = extract_def_use(cfg_cpg, node_successors[-1])
    uses = uses + _t_use + _t_unknown + _t_def

    return [defs, uses, unknown]

# binary_expression
def ddg_binary_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse binary expression and extract def use information. Specifically, binary expression only has used variables.
    """
    # situation: 3 children
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        logger.error('Binary Expression has {} children, cannot handle it' .format(len(node_successors)))
        exit(-1)
    # first child
    _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
    uses = uses + _f_def + _f_use + _f_unknown
    # last child
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])
    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# instanceof_expression
def ddg_instanceof_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse instanceof expression and extract def use information. Specifically, instanceof expression only has used variables.
    """
    defs, uses, unknown = [], [], []

    instanceof_successors = list(cfg_cpg.successors(node))
    if len(instanceof_successors) != 3:
        logger.error('Instanceof Expression has {} children, cannot handle it' .format(len(instanceof_successors)))
        exit(-1)
    # first child
    _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, instanceof_successors[0])
    uses = uses + _f_def + _f_use + _f_unknown
    # last child
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, instanceof_successors[-1])
    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]


# lambda_expression
# TODO cannot support lambda expression now

# ternary_expression
def ddg_ternary_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse ternary expression and extract def use information. Specifically, ternary expression only has used variables.
    """
    defs, uses, unknown = [], [], []

    ternary_successors = list(cfg_cpg.successors(node))

    if len(ternary_successors) != 3:
        logger.error('Ternary Expression has {} children, cannot handle it' .format(len(ternary_successors)))
        exit(-1)
    
    for _successor in ternary_successors:
        if cfg_cpg.nodes[_successor]['cpg_node']:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _s_def + _s_use + _s_unknown
    
    return [defs, uses, unknown]
    


# update_expression
def ddg_update_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse update expression and extract def use information. Specifically, update expression only has def variables.
    """

    defs, uses, unknown = [], [], []
    # update expression has two children
    update_expression_successors = list(cfg_cpg.successors(node))

    if len(update_expression_successors) != 2:
        logger.error('Update expression has {} children, exit' .format(len(update_expression_successors)))
        exit(-1)
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, update_expression_successors[0])
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, update_expression_successors[-1])

    defs = uses + _l_def + _l_use + _l_unknown + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# unary_expression
def ddg_unary_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse unary expression and extract def use information, specifically, unary expression only has use variables.
    """
    defs, uses, unknown = [], [], []

    unary_successors = list(cfg_cpg.successors(node))
    if len(unary_successors) != 2:
        logger.error('Unary Expression has {} children, cannot handle it.' .format(len(unary_successors)))
        exit(-1)
    _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, unary_successors[-1])

    uses = uses + _s_def + _s_use + _s_unknown

    return [defs, uses, unknown]

# cast_expression
def ddg_cast_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse cast expression and extract def-use information, specifically, cast_expression only has use variables.
    """
    defs, uses, unknown = [], [], []
    cast_successors = list(cfg_cpg.successors(node))

    for _successor in cast_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _s_def + _s_use + _s_unknown

    return [defs, uses, unknown]
"""
handle other situations
"""
def ddg_variable_declarator(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse variable declarator and extract def use information. Specifically, variable declarator has both def and use variable.
    """
    defs, uses, unknown = [], [], []
    # variable declarator has 3 or 4 children, 4 has one more dimensions node
    node_successors = list(cfg_cpg.successors(node))
    
    if len(node_successors) not in [1,2, 3, 4]:
        logger.error('Variable Declarator has {} children, exit.' .format(len(node_successors)))
        exit(-1)
    if len(node_successors) == 1:
        _f_defs, _f_uses, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
        defs = defs + _f_defs + _f_unknown
    else:
        _f_defs, _f_uses, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
        _t_defs, _t_uses, _t_unknown = extract_def_use(cfg_cpg, node_successors[-1])
        defs = defs + _f_defs + _f_unknown
        uses = uses + _f_uses + _t_defs + _t_uses + _t_unknown

    return [defs, uses, unknown]

def ddg_argument_list(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse argument list and extract def use information. Specifically, argument list only has used variables.
    """
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _s_def + _s_use + _s_unknown
    
    return [defs, uses, unknown]

def ddg_formal_parameter(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse formal parameter and extract def use information. Specifically, formal parameter only has used variables.
    """
    defs, uses, unknown = [], [], []
    formal_parameter_successors = list(cfg_cpg.successors(node))

    for _successor in formal_parameter_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        defs = defs + _s_def + _s_unknown

    return [defs, uses, unknown]

def ddg_parenthesized_expression(cfg_cpg: DiGraph = None, node: str = None) -> list:
    """Parse parenthesized expression and extract def use information. Specifically, parenthesized expression only has used variables.
    """
    defs, uses, unknown = [], [], []

    parenthesized_successors = list(cfg_cpg.successors(node))

    if len(parenthesized_successors) != 1:
        logger.error('Parenthesize Expression have {} children, Exit'. format(len(parenthesized_successors)))
        exit(-1)
    _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, parenthesized_successors[0])

    uses = _s_def + _s_use + _s_unknown

    return [defs, uses, unknown]
