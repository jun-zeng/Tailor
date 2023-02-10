import os
from graphviz import Digraph
from networkx import DiGraph as nxDiGraph

def gen_dot_graph(node_dict: dict = {}, edge_list: list = [], dot_name: str = None) -> bool:
    """ Generate dot graph according to Nodes and Edges
    
    attributes:
        node_dict -- ASTNode dict.
        edge_list -- Edge list.
        dot_name -- name of dot file.
    
    returns:
        True / False -- generate successfully or not.
    """

    if len(node_dict) == 0:
        print('No nodes.')
        return
    
    dot = Digraph(name = dot_name, format='svg')
    for key, node in node_dict.items():
        if node.node_token:
            label = str(node.node_type + ': ' + node.node_token)
        else:
            label = str(node.node_type)
        dot.node(name = str(key), label = label)
    for edge in edge_list:
        dot.edge(str(edge.start), str(edge.end))
    
    dot_file = os.path.join(os.getcwd(), '{}.gv'.format(dot_name))

    dot.render(filename=dot_file, view=False)
    command = 'rsvg-convert -f pdf -o ' + dot_name + '.pdf ' + dot_file + '.svg; rm ' + dot_file + '.svg'
    os.system(command)
    return True

def visualize_ast_cpg(ast_cpg: nxDiGraph = None, dot_name: str = None, ast_display: bool = True) -> bool:
    """Visualize ast cpg.

    attributes:
        ast_cpg -- a nx.DiGraph instance.
        dot_name -- the name of dot file.
    
    returns:
        True / False -- generate successfully or not.
    """
    nodes = ast_cpg.nodes
    edges = ast_cpg.edges
    dot = Digraph(name = dot_name, format='svg')
    for node in nodes:
        info = ast_cpg.nodes[node]['cpg_node']
        if info.node_token:
            label = str(info.node_type + ': ' + info.node_token)
        else:
            label = str(info.node_type)
        # determine what nodes put in graph
        if ast_display:
            if info.match_statement:
                color = 'Red'
            else:
                color = 'Black'
            dot.node(name=str(node), label=label, color=color)
        else:
            if info.match_statement:
                dot.node(name = str(node), label=label)
    
    for edge in edges:
        edge_type = ast_cpg[edge[0]][edge[1]]['edge_type']
        if edge_type == '100':
            edge_type = None
        if ast_display:
            if edge_type in ['010', '001', '110', '101', '011']:
                dot.edge(str(edge[0]), str(edge[1]), label=edge_type, color='Green')
            else:
                dot.edge(str(edge[0]), str(edge[1]), label=edge_type)
        else:
            if edge_type in ['010', '001', '110', '101', '011']:
                dot.edge(str(edge[0]), str(edge[1]), label=edge_type, color='Green')
    
    dot_file = os.path.join(os.getcwd(), '{}'.format(dot_name))
    dot.render(filename=dot_file, view=False, cleanup=True, format='pdf')
    
    return True