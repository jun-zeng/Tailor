"""Given an instance of Code Property Graph and provide some query apis.
"""
from networkx import DiGraph
from utils.data_structure import Queue
from utils.setting import logger

def entity_query(ast_cpg: DiGraph = None) -> list:
    """Query entity from ast_cpg. Entity is a tuple (hashkey, identifier, token). If identifier or token are empty, it will be None.

    attributes:
        ast_cpg -- an instance of Code Property Graph.
    
    returns:
        entities -- list of entities.
    """
    if ast_cpg == None:
        logger.info('NIL AST Code Property Graph.')
        return False
    
    entities = list()
    ast_nodes = list(ast_cpg.nodes)
    for node in ast_nodes:
        cpg_node = ast_cpg.nodes[node]['cpg_node']
        entity = (cpg_node.node_key, cpg_node.node_type, cpg_node.node_token, cpg_node.match_statement)
        entities.append(entity)
    
    return entities

def edge_query(ast_cpg: DiGraph = None) -> list:
    """Given a cpg instance, query all its edges. Edge is a tuple (head, tail, edge_type) (234, 432, 100).

    attributes:
        ast_cpg -- an instance of Code Property Graph.
    
    returns:
        edges -- list of edge.
    """
    if ast_cpg == None:
        logger.info('NIL AST Code Property Graph.')
        return False
    
    edges = list()
    ast_edges = list(ast_cpg.edges)
    for _e in ast_edges:
        start, end = _e
        edge_type = ast_cpg[start][end]['edge_type']
        edge = (start, end, edge_type)
        edges.append(edge)
    
    return edges

def statentity_query(ast_cpg: DiGraph = None) -> list:
    """Query statement node from ast_cpg.

    attributes:
        ast_cpg -- an instance of Code Property Graph.

    returns:
        stats -- list of statement node identifiers.
    """
    if ast_cpg == None:
        logger.info('NIL AST Code Property Graph.')
        return False
    
    stats = list()
    cpg_nodes = list(ast_cpg.nodes)

    for node in cpg_nodes:
        cpg_node = ast_cpg.nodes[node]['cpg_node']
        if cpg_node.match_statement:
            stats.append(cpg_node.node_key)
    
    return stats

def statnodes_query(ast_cpg: DiGraph = None, node: str = None) -> list:
    """Given a cpg instance, query statement nodes.

    attributes:
        ast_cpg -- an instance of Code Property Graph.
        node -- the identifier of statement node.
    
    returns:
        child_entities -- the ast node children (hashkey).
    """
    child_entities = list()

    queue = Queue()
    queue.push(node)

    while not queue.is_empty():
        current_node = queue.pop()
        node_successors = list(ast_cpg.successors(current_node))
        for _successor in node_successors:
            if not ast_cpg.nodes[_successor]['cpg_node'].match_statement:
                child_entities.append(_successor)
                queue.push(_successor)
    
    return child_entities

def stat_entities(ast_cpg: DiGraph = None) -> list:
    """Find statement's entities.

    attributes:
        ast_cpg -- an instance of Code Property Graph.
    
    returns:
        s_es -- list of statement and its entities.
    """
    s_es = list()

    stats = statentity_query(ast_cpg)

    for stat in stats:
        child_entities = statnodes_query(ast_cpg, stat)
        s_e = [stat, stat] + child_entities
        s_es.append(s_e)
    
    return s_es
