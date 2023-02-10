from networkx import DiGraph
from utils.data_structure import Queue
from utils.setting import logger
from ccpg.util.common import cpg_edge_type

"""
Statements
labeled_statement
compound_statement
expression_statement
if_statement
switch_statement
do_statement
while_statement
for_statement
return_statement
break_statement
continue_statement
goto_statement

declaration
"""

def cfg_build(ast_cpg: DiGraph = None, node: str = None) -> list:
    """Handler of Control Flow Graph builder.

    attributes:
        ast_cpg -- SAST instance (graph representation) for one function.
        node -- identifier of on CPGNode in ast_cpg
    
    returns:
        [entrynode, fringe] -- list including the entrynode and fringe nodes.
    """
    node_type = ast_cpg.nodes[node]['cpg_node'].node_type
    if node_type == 'function_definition':
        entrynode, fringe = cfg_functiondefinition(ast_cpg, node)
    elif node_type == 'labeled_statement':
        entrynode, fringe = cfg_labeledstatement(ast_cpg, node)
    elif node_type == 'compound_statement':
        entrynode, fringe = cfg_compoundstatement(ast_cpg, node)
    elif node_type == 'expression_statement':
        entrynode, fringe = cfg_expressionstatement(ast_cpg, node)
    elif node_type == 'if_statement':
        entrynode, fringe = cfg_ifstatement(ast_cpg, node)
    elif node_type == 'switch_statement':
        entrynode, fringe = cfg_switchstatement(ast_cpg, node)
    elif node_type == 'do_statement':
        entrynode, fringe = cfg_dostatement(ast_cpg, node)
    elif node_type == 'while_statement':
        entrynode, fringe = cfg_whilestatement(ast_cpg, node)
    elif node_type == 'for_statement':
        entrynode, fringe = cfg_forstatement(ast_cpg, node)
    elif node_type == 'return_statement':
        entrynode, fringe = cfg_returnstatement(ast_cpg, node)
    elif node_type == 'break_statement':
        entrynode, fringe = cfg_breakstatement(ast_cpg, node)
    elif node_type == 'continue_statement':
        entrynode, fringe = cfg_continuestatement(ast_cpg, node)
    elif node_type == 'goto_statement':
        entrynode, fringe = cfg_gotostatement(ast_cpg, node)
    elif node_type == 'declaration':
        entrynode, fringe = cfg_declaration(ast_cpg, node)
    elif node_type == 'ret_type':
        entrynode, fringe = cfg_rettype(ast_cpg, node)
    elif node_type == 'case_statement':
        entrynode, fringe = cfg_casestatement(ast_cpg, node)
    else:
        entrynode, fringe = None, []
    
    return entrynode, fringe

def is_statementnode(ast_cpg: DiGraph, node: str) -> bool:
    """Check one node is statement node or not"""
    statement_node_types = ['function_definition', 'labeled_statement', 'compound_statement', 'expression_statement', 'if_statement', 'switch_statement', 'switch_statement', 'do_statement', 'while_statement', 'for_statement', 'return_statement', 'break_statement', 'continue_statement', 'goto_statement', 'declaration', 'ret_type', 'case_statement']

    return ast_cpg.nodes[node]['cpg_node'].node_type in statement_node_types

def seek_parent(ast_cpg: DiGraph, node: str, node_types: list) -> str:
    """Find specific node whose type matches node type in node_types"""
    queue = Queue()
    queue.push(node)
    visited = list()
    visited.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        predecessors = list(ast_cpg.predecessors(current_node))
        for _predecessor in predecessors:
            _predecessor_type = ast_cpg.nodes[_predecessor]['cpg_node'].node_type
            if _predecessor_type not in node_types and is_statementnode(ast_cpg, _predecessor):
                if _predecessor not in visited:
                    visited.append(_predecessor)
                    queue.push(_predecessor)
                else:
                    logger.debug('Seeks parent appears circle edge.')
            elif _predecessor_type in node_types and is_statementnode(ast_cpg, _predecessor):
                return _predecessor
    
    return False

def seek_statement_sibling(ast_cpg: DiGraph, node: str) -> str:
    """Find the node's next statement sibling"""
    queue = Queue()
    queue.push(node)
    visited = list()
    visited.append(node)

    loop_types = ['for_statement', 'do_statement', 'while_statement', 'switch_statement']

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
            # the logic is for situation nested for, and second for has no sibling
            elif not _has_statement_node and ast_cpg.nodes[_predecessor]['cpg_node'].node_type in loop_types:
                return _predecessor
            else:
                if _predecessor not in visited:
                    visited.append(_predecessor)
                    queue.push(_predecessor)
                else:
                    logger.debug('Seek next statement has circle')
    
    return False

def has_default_case(ast_cpg: DiGraph, node: str) -> bool:
    """Check whether switch statement has default option"""
    has_default = False
    node_successors = list(ast_cpg.successors(node))
    for _successor in node_successors:
        if ast_cpg.nodes[_successor]['cpg_node'].node_type == 'default':
            has_default = True
            break
    return has_default
def has_else_default(ast_cpg: DiGraph, node: str):
    node_successors = ast_cpg.successors(node)
    res_has = False
    for _successor in node_successors:
        if ast_cpg.nodes[_successor]['cpg_node'].node_type in ['else', 'default']:
            res_has = True
            break
    
    return res_has

def seek_labeled_statement(ast_cpg: DiGraph, label: str) -> str:
    """Find the labeled statement with specific label"""
    # traverse all nodes in the graph and find labeled statement whose label token matches label.
    all_nodes = ast_cpg.nodes
    for _node in all_nodes:
        if ast_cpg.nodes[_node]['cpg_node'].node_token == label:
            labeled_statement = list(ast_cpg.predecessors(_node))
            # only one parent
            if len(labeled_statement) == 1:
                return labeled_statement[0]
    
    return False

def cfg_singlenode(ast_cpg: DiGraph, node: str) -> list:
    """Singlenode control flow graph"""
    entrynode = node
    fringe = [node]

    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    return [entrynode, fringe]

def cfg_functiondefinition(ast_cpg: DiGraph, node: str) -> list:
    """Parse function definition"""
    entrynode = node
    fringe = [node]

    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    function_successors = list(ast_cpg.successors(node))

    for _successor in function_successors:
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        for _entrynode in fringe:
            edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
            ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type=edge_type)
        fringe = _s_fringe
    
    return [entrynode, fringe]

def cfg_labeledstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse labeled statement"""
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    labeled_statement_successors = list(ast_cpg.successors(node))

    _s_entrynode, fringe = cfg_build(ast_cpg, labeled_statement_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, _s_entrynode, '010')
    ast_cpg.add_edge(entrynode, _s_entrynode, edge_type=edge_type)

    return [entrynode, fringe]

def cfg_compoundstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse compound statement"""
    entrynode = node
    fringe = [node]
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    compound_successors = list(ast_cpg.successors(node))

    for _successor in compound_successors:
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        # to determine whether the node is case statement
        _s_entrynode_type = ast_cpg.nodes[_s_entrynode]['cpg_node'].node_type
        if _s_entrynode_type == 'case_statement':
            for _entrynode in fringe:
                edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
                ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type=edge_type)
            # assume default is the last case (if there is default)
            if has_default_case(ast_cpg, _s_entrynode):
                fringe = _s_fringe
            else:
                fringe = _s_fringe + [node]
        else:
            for _entrynode in fringe:
                edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
                ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type=edge_type)
            fringe = _s_fringe
    
    return [entrynode, fringe]

def cfg_expressionstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse expression statement"""
    return cfg_singlenode(ast_cpg, node)

def cfg_ifstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse if statement"""
    entrynode = node
    fringe = list()
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    if_statement_successors = list(ast_cpg.successors(node))
    c_child = len(if_statement_successors)

    # situation: only have then block
    if c_child == 3:
        then_entrynode, then_fringe = cfg_build(ast_cpg, if_statement_successors[2])
        edge_type = cpg_edge_type(ast_cpg, entrynode, then_entrynode, '010')
        ast_cpg.add_edge(entrynode, then_entrynode, edge_type=edge_type)
        fringe = then_fringe + [node]
    # situation: have both then and else block
    elif c_child == 5:
        then_entrynode, then_fringe = cfg_build(ast_cpg, if_statement_successors[2])
        else_entrynode, else_fringe = cfg_build(ast_cpg, if_statement_successors[4])
        then_edge = cpg_edge_type(ast_cpg, entrynode, then_entrynode, '010')
        else_edge = cpg_edge_type(ast_cpg, entrynode, else_entrynode, '010')
        ast_cpg.add_edge(entrynode, then_entrynode, edge_type=then_edge)
        ast_cpg.add_edge(entrynode, else_entrynode, edge_type=else_edge)
        fringe = then_fringe + else_fringe
    else:
        logger.error('If statement has {} children, only support 3 and 5 children' .format(c_child))
        exit(-1)
    
    return [entrynode, fringe]

def cfg_switchstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse switch statement"""
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    switch_statement_successors = list(ast_cpg.successors(node))

    # handle switch block
    sub_entrynode, fringe = cfg_build(ast_cpg, switch_statement_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type=edge_type)

    return [entrynode, fringe]

def cfg_dostatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse do statement"""
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    do_statement_successors = list(ast_cpg.successors(node))

    # default do statement should have 4 children
    if len(do_statement_successors) != 4:
        logger.error('Do statement has {} children, only support 4 children' .format(len(do_statement_successors)))
        exit(-1)
    # the second child is the body
    sub_entrynode, fringe = cfg_build(ast_cpg, do_statement_successors[1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type=edge_type)

    # add do while loop return edge
    for _fringe in fringe:
        edge_type = cpg_edge_type(ast_cpg, _fringe, entrynode, '010')
        ast_cpg.add_edge(_fringe, entrynode, edge_type=edge_type)
    
    return [entrynode, fringe]

def cfg_whilestatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse while statement"""
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    while_statement_successors = list(ast_cpg.successors(node))

    # the last child is the body
    sub_entrynode, fringe = cfg_build(ast_cpg, while_statement_successors[-1])

    if sub_entrynode:
        edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
        ast_cpg.add_edge(entrynode, sub_entrynode, edge_type=edge_type)
    
    # add while loop return edge
    for _fringe in fringe:
        edge_type = cpg_edge_type(ast_cpg, _fringe, entrynode, '010')
        ast_cpg.add_edge(_fringe, entrynode, edge_type=edge_type)
    
    return [entrynode, fringe + [node]]

def cfg_forstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse for statement"""
    entrynode = node
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()
    for_statement_successors = list(ast_cpg.successors(node))

    # the last children is the for body
    sub_entrynode, fringe = cfg_build(ast_cpg, for_statement_successors[-1])
    edge_type = cpg_edge_type(ast_cpg, entrynode, sub_entrynode, '010')
    ast_cpg.add_edge(entrynode, sub_entrynode, edge_type=edge_type)

    # add for loop return edge
    for _fringe in fringe:
        edge_type = cpg_edge_type(ast_cpg, _fringe, entrynode, '010')
        ast_cpg.add_edge(_fringe, entrynode, edge_type=edge_type)
    
    return [entrynode, fringe + [node]]

def cfg_returnstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse return statement"""
    entrynode = node
    fringe = list()
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    _method = list(ast_cpg.nodes)[0]
    ret_type = list(ast_cpg.successors(_method))[-1]

    # add edge between return statement and ret type
    edge_type = cpg_edge_type(ast_cpg, entrynode, ret_type, '010')
    ast_cpg.add_edge(entrynode, ret_type, edge_type=edge_type)

    return [entrynode, fringe]
    

def cfg_breakstatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse break statement"""
    # In C, break statement has no label. 
    # Just find next statement, and add control flow edge
    entrynode = node
    fringe = list()
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    break_statement_successors = list(ast_cpg.successors(node))
    successors_num = len(break_statement_successors)

    if successors_num != 1:
        logger.error('Break Statement has {} children, only support one child' .format(successors_num))
        exit(-1)
    # situation: break is guarded by if_statement, switch_statement under lopp
    # need connect if_statement, switch_statement to next sibling
    guarded_types = ['if_statement', 'switch_statement']
    parent_node = seek_parent(ast_cpg, node, guarded_types)
    # loop_types = ['for_statement', 'while_statement', 'do_statement']
    if parent_node:
        loop_node = seek_statement_sibling(ast_cpg, parent_node)
        if loop_node:
            loop_guard_type = cpg_edge_type(ast_cpg, parent_node, loop_node, '010')
            ast_cpg.add_edge(parent_node, loop_node, edge_type=loop_guard_type)
    
    # break to next statement
    for_next_types = ['for_statement', 'do_statement', 'while_statement', 'switch_statement']
    _predecessor = seek_parent(ast_cpg, node, for_next_types)

    # situation: cannot find closest loop or switch
    if not _predecessor:
        return [entrynode, fringe]
    
    # find next sibling
    next_statement_node = seek_statement_sibling(ast_cpg, _predecessor)
    if not next_statement_node:
        return [entrynode, fringe]
    else:
        edge_type = cpg_edge_type(ast_cpg, node, next_statement_node, '010')
        ast_cpg.add_edge(node, next_statement_node, edge_type=edge_type)
    
    return [entrynode, fringe]

def cfg_continuestatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse continue statement"""
    entrynode = node
    fringe = list()
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    loop_types = ['for_statement', 'do_statement', 'while_statement']
    loop_node = seek_parent(ast_cpg, node, loop_types)

    if loop_node:
        # add edge from continue statement to loop node
        edge_type = cpg_edge_type(ast_cpg, node, loop_node, '010')
        ast_cpg.add_edge(node, loop_node, edge_type=edge_type)
    
    parent_types = ['if_statement', 'switch_statement']
    parent_node = seek_parent(ast_cpg, node, parent_types)
    if parent_node and not has_else_default(ast_cpg, parent_node):
        fringe = [parent_node]
    
    return [entrynode, fringe]
    
def cfg_gotostatement(ast_cpg: DiGraph, node: str) -> list:
    """"Parse goto statement"""
    # goto statement has no fringe, connect it to the labeled statement.

    entrynode = node
    fringe = list()
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    # goto statement should have 2 children, the last child is label
    goto_statement_successors = list(ast_cpg.successors(node))
    if len(goto_statement_successors) != 2:
        logger.error('Goto statement has {} children, only support 2 children' .format(len(goto_statement_successors)))
        exit(-1)
    
    # connect goto's parent to next sibling
    guarded_types = ['if_statement', 'switch_statement']
    guarded_node = seek_parent(ast_cpg, node, guarded_types)

    if guarded_node and not has_else_default(ast_cpg, guarded_node):
        next_sibling = seek_statement_sibling(ast_cpg, guarded_node)
        if next_sibling:
            _edge_type = cpg_edge_type(ast_cpg, guarded_node, next_sibling, '010')
            ast_cpg.add_edge(guarded_node, next_sibling, edge_type=_edge_type)

    label = ast_cpg.nodes[goto_statement_successors[-1]]['cpg_node'].node_token
    labeled_statement = seek_labeled_statement(ast_cpg, label)
    if labeled_statement:
        edge_type = cpg_edge_type(ast_cpg, node, labeled_statement, '010')
        ast_cpg.add_edge(node, labeled_statement, edge_type=edge_type)
    
    return [entrynode, fringe]

def cfg_declaration(ast_cpg: DiGraph, node: str) -> list:
    """Parse declaration"""
    return cfg_singlenode(ast_cpg, node)

def cfg_rettype(ast_cpg: DiGraph, node: str) -> list:
    return cfg_singlenode(ast_cpg, node)

def cfg_casestatement(ast_cpg: DiGraph, node: str) -> list:
    """Parse case statement"""
    entrynode = node
    fringe = [node]
    ast_cpg.nodes[node]['cpg_node'].set_statement_node()

    case_successors = list(ast_cpg.successors(node))

    for _successor in case_successors:
        _s_entrynode, _s_fringe = cfg_build(ast_cpg, _successor)
        if _s_entrynode == None:
            continue
        for _entrynode in fringe:
            edge_type = cpg_edge_type(ast_cpg, _entrynode, _s_entrynode, '010')
            ast_cpg.add_edge(_entrynode, _s_entrynode, edge_type=edge_type)
        
        fringe = _s_fringe
    
    return [entrynode, fringe]