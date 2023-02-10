from networkx import DiGraph
from utils.setting import logger

def cpg_edge_type(cpg: DiGraph = None, start: str = None, end: str = None, insert_type: str = None) -> str:
    """Generate edge type for Code Property Graph.
    - ast -- 100
    - cfg -- 010
    - pdg -- 001
    - ast & cfg -- 110
    - ast & pdg -- 101
    - cfg & pdg -- 011
    - ast & cfg & pdg -- 111
    attributes
        cpg -- the instance of Code Property Graph.
        start -- the identifier of start node.
        end -- the identifier of end node.
        insert_type -- the edge type waiting to insert.
    
    returns:
        edge_type -- the encoding of edge.
    """
    if cpg == None or start == None or end == None or insert_type == None:
        logger.error('Lack params for generating edge type.')
        exit(-1)
    
    if insert_type == '100':
        edge_type = '100'
    elif insert_type == '010':
        if cpg.has_edge(start, end) and cpg[start][end]['edge_type'] in ['100', '110']:
            edge_type = '110'
        else:
            edge_type = '010'
    elif insert_type == '001':
        if cpg.has_edge(start, end) and cpg[start][end]['edge_type'] in ['100', '101']:
            edge_type = '101'
        elif cpg.has_edge(start, end) and cpg[start][end]['edge_type'] in ['010', '011']:
            edge_type = '011'
        else:
            edge_type = '001'
    else:
        edge_type = '000'
    
    return edge_type

def filter_func(ast_cpg: DiGraph = None, type: str = None) -> bool:
    """Filter functions contain specific type.

    attributes:
        ast_cpg -- an instance of ast_cpg.
        type -- specific type we want to check.

    returns:
        True / False -- if this ast_cpg contains type, return True, else False.
    """
    nodes = list(ast_cpg.nodes)
    for node in nodes:
        if ast_cpg.nodes[node]['cpg_node'].node_type == type:
            return True
    
    return False
