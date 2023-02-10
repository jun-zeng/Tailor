from networkx import DiGraph
from utils.setting import logger
from utils.data_structure import Queue
from ccpg.util.common import cpg_edge_type
from ccpg.cpg.ddg_node import DDGNode


def ddg_build(cfg_cpg: DiGraph, node: str):
    """Identify def-use information in control-flow based code property graph and generate data flow edge for cfg_cpg to form complete code property graph.
    """
    def_use_chain = gen_def_use_chain(cfg_cpg, node)
    for key, value in def_use_chain.items():
        if value.uses == []:
            continue
        for use_var in value.uses:
            back_tracking(cfg_cpg, key, def_use_chain, use_var)

def back_tracking(cfg_cpg: DiGraph, node: str, def_use_chain: dict, use_var: str) -> bool:
    """Given cfg-based code property graph and a node identifier, then do back tracking to add data flow edge
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
                    # logger.error(cfg_cpg.nodes[current_node]['cpg_node'].node_type)
                    # logger.error(cfg_cpg.nodes[_predecessor]['cpg_node'].node_type)
                    # logger.error(_predecessor)
                    logger.error('Current node cannot be find in def use chain, exit')
                    exit(-1)
                ddg_predecessor = def_use_chain[_predecessor]
                if has_dd_rel(use_var, ddg_predecessor):
                    add_ddg_edge(cfg_cpg, ddg_predecessor.node_key, node)
                else:
                    queue.push(_predecessor)
    
    return True

def has_dd_rel(use_var: str, def_node: DDGNode) -> bool:
    """Check whether use_var is defined by def_node
    """
    if use_var == None or def_node == None:
        logger.error('use variable or def node is none, exit')
        exit(-1)
    defs = def_node.get_defs()
    _dd_rel = False
    if use_var in defs:
        _dd_rel = True
    
    return _dd_rel

def add_ddg_edge(cfg_cpg: DiGraph, start_node_identifier: str, end_node_identifier: str) -> bool:
    """Add data flow edge into cfg_cpg to form the complete code property graph
    """
    edge_type = cpg_edge_type(cfg_cpg, start_node_identifier, end_node_identifier, '001')
    cfg_cpg.add_edge(start_node_identifier, end_node_identifier, edge_type=edge_type)

    return True

def check_edge(cfg_cpg: DiGraph, start: str, end: str) -> str:
    """Check the encoding of edge between start and end.
    """
    if cfg_cpg.has_edge(start, end):
        return cfg_cpg[start][end]['edge_type']
    
    return False

def gen_def_use_chain(cfg_cpg: DiGraph, node: str) -> dict:
    """Traverse cfg_cpg and extract def-use information for each statement node in order. Note, we traverse the graph by level-order.
    """
    ddg_chain = dict()

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
                # logger.error(_successor)
                queue.push(_successor)
                check_exist.append(_successor)
    
    return ddg_chain

def merge_def_use(cfg_cpg: DiGraph, node: str) -> DDGNode:
    """Parse one statement node, and traverse its all child entities (non-cfg) to collect its def use information and merge them together.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        defs += _c_def
        uses += _c_use
        unknown += _c_unknown

    # further parse special statements, since they cannot handle in general logic (e.g., return statement)
    node_type = cfg_cpg.nodes[node]['cpg_node'].node_type
    if node_type == 'return_statement':
        uses = uses + unknown
        unknown = []
    elif node_type == 'declaration':
        defs = defs + unknown + uses
        unknown = []
    
    tmp_defs = list(set(defs))
    tmp_uses = list(set(uses))
    tmp_unknown = list(set(unknown))

    tmp_defs.sort(key=defs.index)
    tmp_uses.sort(key=uses.index)
    tmp_unknown.sort(key=unknown.index)


    return DDGNode(node, cfg_cpg.nodes[node]['cpg_node'].node_type, tmp_defs, tmp_uses, tmp_unknown)
    
def extract_def_use(cfg_cpg: DiGraph, node: str) -> list:
    """Traverse entities of statement identified by node to extract its def-use information.
    """
    node_type = cfg_cpg.nodes[node]['cpg_node'].node_type

    if node_type == 'conditional_expression':
        defs, uses, unknown = ddg_conditional_expression(cfg_cpg, node)
    elif node_type == 'assignment_expression':
        defs, uses, unknown = ddg_assignment_expression(cfg_cpg, node)
    elif node_type == 'binary_expression':
        defs, uses, unknown = ddg_binary_expression(cfg_cpg, node)
    elif node_type == 'unary_expression':
        defs, uses, unknown = ddg_unary_expression(cfg_cpg, node)
    elif node_type == 'update_expression':
        defs, uses, unknown = ddg_update_expression(cfg_cpg, node)
    elif node_type == 'cast_expression':
        defs, uses, unknown = ddg_cast_expression(cfg_cpg, node)
    elif node_type == 'pointer_expression':
        defs, uses, unknown = ddg_pointer_expression(cfg_cpg, node)
    elif node_type == 'sizeof_expression':
        defs, uses, unknown = ddg_sizeof_expression(cfg_cpg, node)
    elif node_type == 'subscript_expression':
        defs, uses, unknown = ddg_subscript_expression(cfg_cpg, node)
    elif node_type == 'call_expression':
        defs, uses, unknown = ddg_call_expression(cfg_cpg, node)
    elif node_type == 'field_expression':
        defs, uses, unknown = ddg_field_expression(cfg_cpg, node)
    elif node_type == 'compound_literal_expression':
        defs, uses, unknown = ddg_compound_literal_expression(cfg_cpg, node)
    elif node_type == 'identifier':
        defs, uses, unknown = ddg_identifier(cfg_cpg, node)
    elif node_type == 'argument_list':
        defs, uses, unknown = ddg_argument_list(cfg_cpg, node)
    elif node_type == 'init_declarator':
        defs, uses, unknown = ddg_init_declarator(cfg_cpg, node)
    elif node_type == 'initializer_list':
        defs, uses, unknown = ddg_initializer_list(cfg_cpg, node)
    elif node_type == 'parameter_declaration':
        defs, uses, unknown = ddg_parameter_declaration(cfg_cpg, node)
    elif node_type == 'declaration':
        defs, uses, unknown = ddg_declaration(cfg_cpg, node)
    elif node_type == 'array_declarator':
        defs, uses, unknown = ddg_array_declarator(cfg_cpg, node)
    else:
        defs, uses, unknown = ddg_deeper(cfg_cpg, node)
    
    return [defs, uses, unknown]

def ddg_deeper(cfg_cpg: DiGraph, node: str) -> list:
    """Parse node not be expression"""
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    # Traverse its children 
    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _successor)
        defs += _s_def
        uses += _s_use
        unknown += _s_unknown
    
    return [defs, uses, unknown]

def combine_fields(left_v: list, right_v: list) -> list:
    """Combine two variable list and generate new def use variable.
    
    left_v use, left+right def
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

"""
Handle Expressions
"""
# conditional_expression
def ddg_conditional_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse conditional expression and extract def use information
    
    Three children: condition, consequence, alternative
    All these parts belong to uses.
    """
    defs, uses, unknown = [], [], []

    # condition expression should have three children.
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        logger.error('Conditional expression has {} children, only support 3 children.' .format(len(node_successors)))
        exit(-1)
    
    for _successor in node_successors:
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _c_def + _c_use + _c_unknown
    
    return [defs, uses, unknown]

# assignment_expression
def ddg_assignment_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse assignment expression and extract def use information
    
    Left part of assignment expression belongs to defs.
    Right part of assignment expression belongs to uses.
    """
    defs, uses, unknown = [], [], []

    # assignment expression should have three children
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        logger.error('Assignment expression has {} children, only support 3 children.' .format(len(node_successors)))
        exit(-1)
    
    # parse left part
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, node_successors[0])
    defs = defs + _l_def + _l_unknown
    uses = uses + _l_use

    # parse right part
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])
    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]


# binary_expression
def ddg_binary_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse binary expression and extract def use information
    
    Three children: left, right --> identifier, middle --> operator
    Left and Right parts both belong to uses.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        logger.error('Binary expression has {} children, only support 3 children'. format(len(node_successors)))
        exit(-1)
    
    # parse left part
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, node_successors[0])

    # parse right part
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    uses = uses + _l_def + _l_use + _l_unknown + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# unary_expression
def ddg_unary_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse unary expression and extract def use information.

    Two children: operator & argument
    Both children belong to uses.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 2:
        logger.error('Unary expression has {} children, only support 2 children' .format(len(node_successors)))
        exit(-1)
    
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# update_expression
def ddg_update_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse identifier and extract def use information.

    Two children argument & operator:
    [operator, argument] [argument, operator]
    """
    defs, uses, unknown = [], [], []

    update_expression_successors = list(cfg_cpg.successors(node))

    if len(update_expression_successors) != 2:
        logger.error('Update expression has {} children, only support 2 children.' .format(len(update_expression_successors)))
        exit(-1)
    
    # parse left children
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, update_expression_successors[0])

    # parse right children
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, update_expression_successors[-1])

    # update expression only has uses variables
    defs = uses + _l_def + _l_use + _l_unknown + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# cast_expression
def ddg_cast_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse cast expression and extract def use information.

    Two children: type_descriptor, expression
    Right child belongs to uses.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 2:
        logger.error('Cast expression has {} children, only support 2 children' .format(len(node_successors)))
        exit(-1)
    
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# pointer_expression
def ddg_pointer_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse pointer expression and extract def use information.
    
    Two children: operator, expression
    Right child belongs to uses.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 2:
        logger.error('Pointer expression has {} children, only support 2 children' .format(len(node_successors)))
        exit(-1)
    
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# sizeof_expression
def ddg_sizeof_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse sizeof expression and extract def use information.
    
    Two children, right child belongs to use.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 2:
        logger.error('Sizeof expression has {} children, only support 2 children' .format(len(node_successors)))
        exit(-1)
    
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# subscript_expression
def ddg_subscript_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse subscript expression and extract def use information.
    
    Two children, both children belongs to uses.
    """
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    if len(node_successors) != 2:
        logger.error('Subscript expression has {} children, only support 2 children.' .format(len(node_successors)))
        exit(-1)
    
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, node_successors[0])
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    uses = uses + _l_def + _l_use + _l_unknown + _r_def + _r_use + _r_unknown
    
    return [defs, uses, unknown]

# call_expression
def ddg_call_expression(cfg_cpg: DiGraph, node: str) -> list:
    """parse call expression and extract def use information.
    
    Two children: call expression only has used variables.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _c_def + _c_use + _c_unknown

    return [defs, uses, unknown]

# field_expression
def ddg_field_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse field expression and extract def use information.
    
    Two children, need combination (a.b.c, use a, a.b, a.b.c)
    """

    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 2:
        logger.error('Field expression has {} children, only support 2 children' .format(len(node_successors)))
        exit(-1)
    
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, node_successors[0])
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    defs = _r_def + _r_use + _r_unknown
    uses = _l_def + _l_use + _r_unknown
    defs = combine_fields(uses, defs)

    return [defs, uses, unknown]

# compound_literal_expression
def ddg_compound_literal_expression(cfg_cpg: DiGraph, node: str) -> list:
    """Parse compound literal expression and extract def use information
    
    Two children, both of them belong to uses.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 2:
        logger.error('Compound literal expression has {} children, only support 2 children')
        exit(-1)
    
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])

    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# identifier
def ddg_identifier(cfg_cpg: DiGraph, node: str) -> list:
    """Parse identifier and extract def use information. 
    
    Identifier cannot reflect its def use information accurately. We set it as unknown by default.
    """
    defs, uses = [], []
    unknown = [cfg_cpg.nodes[node]['cpg_node'].node_token]

    return [defs, uses, unknown]

"""
Other expressions
"""
# argument_list
def ddg_argument_list(cfg_cpg: DiGraph, node: str) -> list:
    """Parse argument list and extract def use information.

    Do not know the number of children. However, all of them belong to uses.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _c_def + _c_use + _c_unknown
    
    return [defs, uses, unknown]

# init_declarator
def ddg_init_declarator(cfg_cpg: DiGraph, node: str) -> list:
    """Parse init declarator and extract def use information.

    Three children
    Left child: def
    Right child: use
    """   
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        logger.error('Init declarator has {} children, only support 3 children' .format(len(node_successors)))
        exit(-1)
    
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, node_successors[0])
    defs = _l_def + _l_unknown

    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])
    uses = _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]

# initializer_list
def ddg_initializer_list(cfg_cpg: DiGraph, node: str) -> list:
    """Parse initializer list
    
    Do not the number of children, however, all of them belong to use.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))

    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _c_def + _c_use + _c_unknown
    
    return [defs, uses, unknown]

# parameter_declaration
def ddg_parameter_declaration(cfg_cpg: DiGraph, node: str) -> list:
    """Parse parameter_declaration and extract def use information.

    All children belong to defs.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))

    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        defs = defs + _c_def + _c_unknown
    
    return [defs, uses, unknown]

# declaration
def ddg_declaration(cfg_cpg: DiGraph, node: str) -> list:
    """Parse declaration and extract def use information.

    Convert all unknown to defs
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        defs = defs + _c_def + _c_unknown
        uses = uses + _c_use
    
    return [defs, uses, unknown]

# array_declarator
def ddg_array_declarator(cfg_cpg: DiGraph, node: str) -> list:
    """Parse array declarator and extract def use information.
    
    All children should belong to uses.
    """
    defs, uses, unknown = [], [], []

    node_successors = list(cfg_cpg.successors(node))
    for _successor in node_successors:
        if cfg_cpg.nodes[_successor]['cpg_node'].match_statement:
            continue
        _c_def, _c_use, _c_unknown = extract_def_use(cfg_cpg, _successor)
        uses = uses + _c_def + _c_use + _c_unknown
    
    return [defs, uses, unknown]