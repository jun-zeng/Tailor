from networkx import DiGraph
from javacpg.util.common import cpg_edge_type
from utils.data_structure import Queue
from utils.setting import logger

def cfg_build(ast_cpg: DiGraph = None, node: str = None) -> list:
    """Handler of Control Flow Graph builder.

    attributes:
        ast_cpg -- SAST instance (graph representation) of one function.
        node -- identifier of one CPGNode in ast_cpg.
    
    returns:
        [entrynode, fringe] -- list including the entrynode and fringe nodes.
    """
    node_type = ast_cpg.nodes[node]['cpg_node'].node_type
    if node_type == 'method_declaration':
        entrynode, fringe = cfg_methoddeclaration(ast_cpg, node)
    elif node_type == 'expression_statement':
        entrynode, fringe = cfg_statementexpression(ast_cpg, node)
    elif node_type == 'local_variable_declaration':
        entrynode, fringe = cfg_localvariabledeclaraion(ast_cpg, node)
    elif node_type == 'if_statement':
        entrynode, fringe = cfg_ifstatement(ast_cpg, node)
    elif node_type == 'for_statement':
        entrynode, fringe = cfg_forstatement(ast_cpg, node)
    elif node_type == 'enhanced_for_statement':
        entrynode, fringe = cfg_forstatement(ast_cpg, node)
    elif node_type == 'while_statement':
        entrynode, fringe = cfg_whilestatement(ast_cpg, node)
    elif node_type == 'do_statement':
        entrynode, fringe = cfg_dowhilestatement(ast_cpg, node)
    elif node_type == 'switch_expression':
        entrynode, fringe = cfg_switchexpression(ast_cpg, node)
    elif node_type == 'switch_block':
        entrynode, fringe = cfg_switchblock(ast_cpg, node)
    elif node_type == 'switch_block_statement_group':
        entrynode, fringe = cfg_switchblockgroup(ast_cpg, node)
    elif node_type == 'return_statement':
        entrynode, fringe = cfg_returnstatement(ast_cpg, node)
    elif node_type == 'block':
        entrynode, fringe = cfg_blockstatement(ast_cpg, node)
    elif node_type == 'ret_type':
        entrynode, fringe = cfg_rettype(ast_cpg, node)
    elif node_type == 'break_statement':
        entrynode, fringe = cfg_breakstatement(ast_cpg, node)
    elif node_type == 'continue_statement':
        entrynode, fringe = cfg_continuestatement(ast_cpg, node)
    elif node_type == 'labeled_statement':
        entrynode, fringe = cfg_labeledstatement(ast_cpg, node)
    elif node_type == 'try_statement':
        entrynode, fringe = cfg_trystatement(ast_cpg, node)
    elif node_type == 'try_with_resources_statement':
        entrynode, fringe = cfg_trystatement(ast_cpg, node)
    elif node_type == 'catch_clause':
        entrynode, fringe = cfg_catchclause(ast_cpg, node)
    elif node_type == 'finally_clause':
        entrynode, fringe = cfg_finallyclause(ast_cpg, node)
    elif node_type == 'synchronized_statement':
        entrynode, fringe = cfg_synchblockstatement(ast_cpg, node)
    elif node_type == 'throw_statement':
        entrynode, fringe = cfg_throwstatement(ast_cpg, node)
    elif node_type == 'assert_statement':
        entrynode, fringe = cfg_assert_statement(ast_cpg, node)
    elif node_type == 'yield_statement':
        entrynode, fringe = cfg_yield_statement(ast_cpg, node)
    else:
        entrynode, fringe = None, []
           
    return entrynode, fringe

def seek_parent(ast_cpg: DiGraph = None, node: str = None, node_types: list = []) -> str:
    """ Find specific node whose type matches node_type.
    """
    # adopt level-order traversal to find the first predecessor whose node type matches node_types
    queue = Queue()
    queue.push(node)
    node_flag = list()
    node_flag.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        predecessors = list(ast_cpg.predecessors(current_node))
        for _predecessor in predecessors:
            _predecessor_type = ast_cpg.nodes[_predecessor]['cpg_node'].node_type
            if  _predecessor_type not in node_types and is_statementnode(ast_cpg, _predecessor):
                if _predecessor not in node_flag:
                    node_flag.append(_predecessor)
                    queue.push(_predecessor)
                else:
                    logger.debug('Circle Edge')
            elif _predecessor_type in node_types and is_statementnode(ast_cpg, _predecessor):
                return _predecessor

    return False

def seek_statement_sibling(ast_cpg: DiGraph = None, node: str = None) -> str:
    """Given a node in ast_cpg, find its next statement sibling.

    attributes:
        ast_cpg -- SAST instance (graph representation) of one function.
        node -- identifier of one CPGNode in ast_cpg.
    
    returns:
        identifier -- the identifier of next statement sibling, if there is no sibling, return False.
    """
    queue = Queue()
    queue.push(node)
    node_flag = list()
    node_flag.append(node)

    loop_types = ['for_statement', 'enhanced_for_statement', 'while_statament', 'do_statement', 'switch_expression']
    # do level-order traversal to find its next statement sibling.
    while not queue.is_empty():
        current_node = queue.pop()
        predecessors = list(ast_cpg.predecessors(current_node))
        for _predecessor in predecessors:
            _p_successors = list(ast_cpg.successors(_predecessor))
            cur_node_idx = _p_successors.index(current_node) + 1
            siblings = _p_successors[cur_node_idx:]
            _has_statement_node = False
            next_statement = None
            for _sibling in siblings:
                if is_statementnode(ast_cpg, _sibling):
                    _has_statement_node = True
                    next_statement = _sibling
                    break
            if _has_statement_node:
                return next_statement
            elif not _has_statement_node and ast_cpg.nodes[_predecessor]['cpg_node'].node_type in loop_types:
                return _predecessor
            else:
                if _predecessor not in node_flag:
                    node_flag.append(_predecessor)
                    queue.push(_predecessor)
                else:
                    logger.debug('Circle Edge')
    
    return False

def seek_labeled_statement(ast_cpg: DiGraph = None, label: str = None) -> str:
    """
    """
    # traverse all nodes in the graph and find labeled_statement whose label token matches label.
    all_nodes = ast_cpg.nodes
    for _node in all_nodes:
        if ast_cpg.nodes[_node]['cpg_node'].node_token == label:
            label_statements = list(ast_cpg.predecessors(_node))
            # only one parent
            if len(label_statements) == 1:
                return label_statements[0]
    
    return False
            
def is_statementnode(ast_cpg: DiGraph = None, node: str = None) -> bool:
    """
    """
    statement_node_types = ['method_declaration', 'expression_statement', 'local_variable_declaration', 'if_statement', 'for_statement', 'while_statement', 'switch_statement', 'switch_block', 'switch_block_statement_group', 'return_statement', 'block', 'ret_type', 'break_statement', 'enhanced_for_statement', 'continue_statement', 'try_statement', 'do_statement', 'switch_expression', 'label_statement', 'catch_clause', 'finally_clause', 'synchronized_statement', 'throw_statement']

    return ast_cpg.nodes[node]['cpg_node'].node_type in statement_node_types

def seek_valid_fringe(ast_cpg: DiGraph = None, node: str = None) -> str:
    """Given a ast cpg instance and one node identifier, find its valid fringe.
    For example, in a break statement is inserted in try block, try block will no fringe, and now we need add a valid fringe for try block for the completness .

    attributes:
        ast_cpg -- an ast_cpg instance.
        node -- the identifier of node waiting to find valid fringe (such as try block).
    
    returns:
        valid_fringe -- valid fringe of current node.
    """
    # reverse node's children
    node_successors = list(ast_cpg.successors(node))
    node_successors.reverse()

    exclude_nodes = ['break_statement', 'continue_statement']

    for _successor in node_successors:
        ast_node = ast_cpg.nodes[_successor]['cpg_node']
        if ast_node.node_type in exclude_nodes:
            continue
        else:
            if is_statementnode(ast_cpg, _successor):
                return _successor
    
    # if find no fringe, the fringe is itself.
    return node  
                    
def cfg_singlenode(ast_cpg: DiGraph = None, node: str = None) -> list:
    """ For CPGNode in ast_cpg, if it matches one statement, then set match_statement as True. 

    attributes:
        ast_cpg -- Code Property Graph constructed by ASTNode.
        node -- identifier used to visit specific node in ast cpg.

    returns:
        [entrynode, fringe] -- list including entrynode and fringe nodes.
    """
    entrynode = node
    fringe = [node]
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    return [entrynode, fringe]

"""
handlers for different kinds of expression statement in java code.
"""
def cfg_methoddeclaration(ast_cpg: DiGraph = None, node: str = None) -> list:
    """Generate method cfg.

    attributes:
        ast_cpg -- Code Property Graph constructed by ASTNode.
        node -- identifier used to visit specific node in ast cpg.

    returns:
        [entrynode, fringe, exitnode] -- list including entrynode, fringe nodes and exitnode.
    """
    entrynode = node
    fringe = [node]

    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    method_successors = list(ast_cpg.successors(node))
    if len(method_successors) == 0:
        return [entrynode, fringe]
    else:
        for _successor in method_successors:
            _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
            if _s_entrynode == None:
                continue
            for _entrynode in fringe:
                edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
                ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type = edge_type)
            fringe = _s_fringe
            # print(ast_cpg.nodes[fringe[0]]['cpg_node'].node_type)
    
    return [entrynode, fringe]

def cfg_rettype(ast_cpg: DiGraph, node: str) -> list:
    return cfg_singlenode(ast_cpg, node)


def cfg_statementexpression(ast_cpg: DiGraph, node: str) -> list:
    return cfg_singlenode(ast_cpg, node)

def cfg_localvariabledeclaraion(ast_cpg: DiGraph, node: str) -> list:
    return cfg_singlenode(ast_cpg, node)

def cfg_assert_statement(ast_cpg: DiGraph, node: str) -> list:
    return cfg_singlenode(ast_cpg, node)

def cfg_yield_statement(ast_cpg: DiGraph, node: str) -> list:
    """ TODO 
    """
    return [None, []]

def cfg_labeledstatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """

    # By default, we think labeled_statement has two children.
    entrynode = node

    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    labeled_statement_successors = list(ast_cpg.successors(node))

    _s_entrynode, fringe = cfg_build(ast_cpg, labeled_statement_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, _s_entrynode, '010')
    ast_cpg.add_edge(entrynode, _s_entrynode, edge_type = edge_type)

    return [entrynode, fringe]

def cfg_throwstatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    return cfg_singlenode(ast_cpg, node)


def cfg_ifstatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    fringe = []
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    if_statement_successors = list(ast_cpg.successors(node))
    c_child = len(if_statement_successors)

    # handle situation: corner case: two statement if ();
    if c_child == 2:
        fringe = [node]
    # handle situation: only have then block.
    elif c_child == 3:
        then_entrynode, then_fringe = cfg_build(ast_cpg, if_statement_successors[2])
        edge_type = cpg_edge_type(ast_cpg, entrynode, then_entrynode, '010')
        ast_cpg.add_edge(entrynode, then_entrynode, edge_type = edge_type)
        fringe = then_fringe + [node]
    # handle situation: have else block
    elif c_child == 4:
        else_entrynode, else_fringe = cfg_build(ast_cpg, if_statement_successors[-1])
        edge_type = cpg_edge_type(ast_cpg, entrynode, else_entrynode, '010')
        ast_cpg.add_edge(entrynode, else_entrynode, edge_type = edge_type)
        fringe = else_fringe
    # handle situation: have both then and else block.
    elif c_child == 5:
        then_entrynode, then_fringe = cfg_build(ast_cpg, if_statement_successors[2])
        else_entrynode, else_fringe = cfg_build(ast_cpg, if_statement_successors[4])
        then_edge = cpg_edge_type(ast_cpg, entrynode, then_entrynode, '010')
        else_edge = cpg_edge_type(ast_cpg, entrynode, else_entrynode, '010')
        ast_cpg.add_edge(entrynode, then_entrynode, edge_type = then_edge)
        ast_cpg.add_edge(entrynode, else_entrynode, edge_type = else_edge)
        fringe = then_fringe + else_fringe
    else:
        logger.error('For if statement, we handle 2, 3 or 5 children, this statement has {} children'.format(c_child))
        exit(-1)
    
    return [entrynode, fringe]
    
def cfg_forstatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    for_statement_successors = list(ast_cpg.successors(node))

    # handle corner case for(;;);
    if not is_statementnode(ast_cpg, for_statement_successors[-1]):
        return [entrynode, [node]]

    # handle situation: regard the last children as waiting parsed node.
    sub_entrynode, fringe = cfg_build(ast_cpg, for_statement_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type = edge_type)

    # add for loop return edge
    for _fringe in fringe:
        edge_type = cpg_edge_type(ast_cpg, _fringe, entrynode, '010')
        ast_cpg.add_edge(_fringe, entrynode, edge_type = edge_type)

    return [entrynode, fringe + [node]]


def cfg_whilestatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    while_statement_successors = list(ast_cpg.successors(node))

    # handle situation: regard the last children as wait parsed node.
    sub_entrynode, fringe = cfg_build(ast_cpg, while_statement_successors[-1])

    if sub_entrynode:
        edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
        ast_cpg.add_edge(entrynode, sub_entrynode, edge_type = edge_type)

    # add while loop return edge
    for _fringe in fringe:
        edge_type = cpg_edge_type(ast_cpg, _fringe, entrynode, '010')
        ast_cpg.add_edge(_fringe, entrynode, edge_type = edge_type)

    return [entrynode, fringe + [node]]



def cfg_dowhilestatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    dowhile_statement_successors = list(ast_cpg.successors(node))

    # default do_statement should have 4 children.
    if len(dowhile_statement_successors) != 4:
        logger.warn('Do While Statement has no 4 children')
        return [entrynode, []]
    
    sub_entrynode, fringe = cfg_build(ast_cpg, dowhile_statement_successors[1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type = edge_type)

    # add do while loop return edge
    for _fringe in fringe:
        edge_type = cpg_edge_type(ast_cpg, _fringe, entrynode, '010')
        ast_cpg.add_edge(_fringe, entrynode, edge_type = edge_type)
    
    return [entrynode, fringe]
    
def cfg_switchexpression(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    switch_expression_successors = list(ast_cpg.successors(node))

    # handle switch block.
    sub_entrynode, fringe = cfg_build(ast_cpg, switch_expression_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type = edge_type)

    return [entrynode, fringe + [node]]

def cfg_switchblock(ast_cpg: DiGraph, node: str) -> str:
    """
    """
    entrynode = node
    fringe = [node]
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    switch_block_successors = list(ast_cpg.successors(node))

    # handle different switch_block_statement_group
    # first add edge between switch_block and switch_block_statement_group
    for _successor in switch_block_successors:
        switch_group_edge = cpg_edge_type(ast_cpg, entrynode, _successor, '010')
        ast_cpg.add_edge(entrynode, _successor, edge_type = switch_group_edge)
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        for _entrynode in fringe:
            edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
            ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type = edge_type)
        fringe = _s_fringe
    
    return [entrynode, fringe]

def cfg_switchblockgroup(ast_cpg: DiGraph, node: str) -> str:
    """
    """
    entrynode = node
    fringe = [node]
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    blockgroup_successors = list(ast_cpg.successors(node))

    for _successor in blockgroup_successors:
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        for _entrynode in fringe:
            edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
            ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type = edge_type)
        
        fringe = _s_fringe
    
    return [entrynode, fringe]

def cfg_returnstatement(ast_cpg: DiGraph, node: str) -> list:
    """For returnstatement, its entrynode is itself, and there is no fringe. We need to connect returnstatement to ret type directly.
    """
    entrynode = node
    fringe = []
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    _method = list(ast_cpg.nodes)[0]
    ret_type = list(ast_cpg.successors(_method))[-1]
    # add edge between return statement and ret type
    edge_type = cpg_edge_type(ast_cpg, entrynode, ret_type, '010')
    ast_cpg.add_edge(entrynode, ret_type, edge_type = edge_type)

    return [entrynode, fringe]


def cfg_breakstatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    # break_statement has no fringe, but we should connect it with the break label.
    # here we consider * kinds of label.
    # 1. loop and switch end node.
    # 2. flag label. (label statement)
    entrynode = node
    fringe = []
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    
    break_statement_successors = list(ast_cpg.successors(node))
    successors_num = len(break_statement_successors)

    # handle situtation: connect break's parent to next sibling
    parent_types = ['if_statement', 'switch_expression']
    parent_node = seek_parent(ast_cpg, node, parent_types)
    # loop_types = ['while_statement', 'do_statement', 'for_statement', 'enhanced_for_statement']
    # TODO check bug
    if parent_node:
        next_sibling = seek_statement_sibling(ast_cpg, parent_node)
        if next_sibling:
            parent_loop_type = cpg_edge_type(ast_cpg, parent_node, next_sibling, '010')
            ast_cpg.add_edge(parent_node, next_sibling, edge_type = parent_loop_type)

    # handle no flag label situation.
    if successors_num == 1:
        for_next_types = ['for_statement', 'while_statement', 'switch_expression', 'enhanced_for_statement', 'do_statement']
        _predecessor = seek_parent(ast_cpg, node, for_next_types)

        # handle situation: cannot find closest loop or switch 
        if not _predecessor:
            return [entrynode, fringe]

        next_statement_node = seek_statement_sibling(ast_cpg, _predecessor)
        if not next_statement_node:
            return [entrynode, fringe]
        else:
            edge_type = cpg_edge_type(ast_cpg, node, next_statement_node, '010')
            ast_cpg.add_edge(node, next_statement_node, edge_type = edge_type)
    
    # handle flag label situation.
    elif successors_num == 2:
        label = ast_cpg.nodes[break_statement_successors[-1]]['cpg_node'].node_token
        labeled_statement = seek_labeled_statement(ast_cpg, label)
        if labeled_statement:
            edge_type = cpg_edge_type(ast_cpg, node, labeled_statement, '010')
            ast_cpg.add_edge(node, labeled_statement, edge_type = edge_type)
    
    return [entrynode, fringe]

def cfg_continuestatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    def has_else_default(ast_cpg: DiGraph, node: str):
        node_successors = ast_cpg.successors(node)
        res_has = False
        for _successor in node_successors:
            if ast_cpg.nodes[_successor]['cpg_node'].node_type in ['else', 'default']:
                res_has = True
                break
        
        return res_has
    entrynode = node
    fringe = []
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    loop_types = ['while_statement', 'do_statement', 'for_statement', 'enhanced_for_statement']
    loop_node = seek_parent(ast_cpg, node, loop_types)

    if loop_node:
        # add edge from continue statement to loop node
        edge_type = cpg_edge_type(ast_cpg, node, loop_node, '010')
        ast_cpg.add_edge(node, loop_node, edge_type = edge_type)
    
    # TODO may have bug
    parent_types = ['if_statement', 'switch_expression']
    parent_node = seek_parent(ast_cpg, node, parent_types)
    if parent_node and not has_else_default(ast_cpg, parent_node):
        fringe = [parent_node]

    return [entrynode, fringe]

def cfg_synchblockstatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    fringe = [node]
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    synchblockstatement_successors = list(ast_cpg.successors(node))

    for _successor in synchblockstatement_successors:
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        for _entrynode in fringe:
            edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
            ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type = edge_type)
        fringe = _s_fringe
    
    return [entrynode, fringe]

def cfg_trystatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    # For try statement, parse its children
    entrynode = node
    fringe = []
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    trystatement_successors = list(ast_cpg.successors(node))

    if len(trystatement_successors) == 0:
        return [entrynode, [node]]

    try_block_fringe = []
    try_catch_fringe = []

    for _successor in trystatement_successors:
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        # to determine whether the node is block 
        _s_entrynode_type = ast_cpg.nodes[_s_entrynode]['cpg_node'].node_type
        if  _s_entrynode_type == 'block':
            # handle situation: block has no fringe (maybe result from break statement)
            if len(_s_fringe) == 0:
                _s_fringe = seek_valid_fringe(ast_cpg, _s_entrynode)
                _s_fringe = [_s_fringe]
            try_block_fringe += _s_fringe
            try_catch_fringe += _s_fringe
            edge_type = cpg_edge_type(ast_cpg, entrynode, _s_entrynode, '010')
            ast_cpg.add_edge(entrynode, _s_entrynode, edge_type = edge_type)
            fringe = try_block_fringe
        elif _s_entrynode_type == 'catch_clause':
            for _entrynode in try_block_fringe:
                edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
                ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type = edge_type)
                try_catch_fringe += _s_fringe
            fringe = try_catch_fringe
        elif _s_entrynode_type == 'finally_clause':
            for _entrynode in try_catch_fringe:
                edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode_type, '010')
                ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type = edge_type)
            fringe = _s_fringe
        else:
            logger.debug('Not implement label. Exit.')
            exit(-1)
    
    return [entrynode, fringe]
             

def cfg_catchclause(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    catchclause_successors = list(ast_cpg.successors(node))

    sub_entrynode, fringe = cfg_build(ast_cpg, catchclause_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type = edge_type)

    return [entrynode, fringe]


def cfg_finallyclause(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    finallyclause_successors = list(ast_cpg.successors(node))

    sub_entrynode, fringe = cfg_build(ast_cpg, finallyclause_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type = edge_type)

    return [entrynode, fringe]

def cfg_blockstatement(ast_cpg: DiGraph, node: str) -> list:
    """
    """
    entrynode = node
    fringe = [node]
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    block_successors = list(ast_cpg.successors(node))
    for _successor in block_successors:
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        for _entrynode in fringe:
            edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
            ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type = edge_type)
        fringe = _s_fringe
    
    return [entrynode, fringe]
