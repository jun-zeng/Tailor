from networkx import DiGraph
from treelib import Tree
from utils.data_structure import Queue
from utils.setting import logger
from javacpg.util.common import cpg_edge_type
from javacpg.cpg.cpg_node import CPGNode


def gen_ast_cpg(sast: Tree = None) -> DiGraph:
    """Transform SAST (simplified Abstract Syntax Tree) to CPG.

    attributes:
        sast -- a SAST instance of one function.
    
    returns:
        ast_cpg -- an instance of AST CPG of current function.
    """
    if sast == None:
        logger.info('NIL SAST. Return.')
        
        return False

    ast_cpg = DiGraph()

    root = sast.root

    # add root node to ast_cpg
    ast_cpg.add_node(root, cpg_node=CPGNode(sast.get_node(root).data))

    queue = Queue()
    queue.push(root)

    while not queue.is_empty():
        current_node = queue.pop()

        for child in sast.children(current_node):
            child_identifier = child.identifier
            child_data = child.data
            cpg_node = CPGNode(child_data)
            ast_cpg.add_node(child_identifier, cpg_node=cpg_node)
            edge_type = cpg_edge_type(ast_cpg, current_node, child_identifier, '100')
            ast_cpg.add_edge(current_node, child_identifier, edge_type = edge_type)
            queue.push(child_identifier)
    
    return ast_cpg
