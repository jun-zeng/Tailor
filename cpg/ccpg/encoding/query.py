from utils.data_structure import Queue
from networkx import DiGraph

def entity_query(cpg: DiGraph) -> list:
    """Query entity from cpg, entity is a tuple (hashkey, identifier, token). If identifier or token are empty, it will be none.
    """
    entities = list()
    ast_nodes = list(cpg.nodes)

    for node in ast_nodes:
        cpg_node = cpg.nodes[node]['cpg_node']
        entity = (cpg_node.node_key, cpg_node.node_type, cpg_node.node_token, cpg_node.match_statement)
        entities.append(entity)
    
    return entities

def edge_query(cpg: DiGraph) -> list:
    """Given a cpg instance, query all edges. Edge is a tuple (head, tail, edge_type)
    """
    edges = list()
    cpg_edges = list(cpg.edges)

    for _e in cpg_edges:
        start, end = _e
        edge_type = cpg[start][end]['edge_type']
        edge = (start, end, edge_type)
        edges.append(edge)
    
    return edges

def statentity_query(cpg: DiGraph) -> list:
    """Query statement node from cpg
    """
    stats = list()
    cpg_nodes = list(cpg.nodes)

    for node in cpg_nodes:
        cpg_node = cpg.nodes[node]['cpg_node']
        if cpg_node.match_statement:
            stats.append(cpg_node.node_key)
    
    return stats

def statnodes_query(cpg: DiGraph, node: str) -> list:
    """Given a cpg instance, statement node, query its attribute nodes
    """
    child_entities = list()

    queue = Queue()
    queue.push(node)

    while not queue.is_empty():
        current_node = queue.pop()
        node_successors = list(cpg.successors(current_node))
        for _successor in node_successors:
            if not cpg.nodes[_successor]['cpg_node'].match_statement:
                child_entities.append(_successor)
                queue.push(_successor)
    
    return child_entities

def stat_entities(cpg: DiGraph) -> list:
    """Find statement's entities
    """

    statement_entities = list()
    stats = statentity_query(cpg)

    for stat in stats:
        child_entities = statnodes_query(cpg, stat)
        s_e = [stat, stat] + child_entities
        statement_entities.append(s_e)
    
    return statement_entities