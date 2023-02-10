# Provide some functions to calculate statistics of cpg.
import numpy as np
from networkx import DiGraph
from utils.setting import logger

def print_cpg_statistics(cpg_dict: dict) -> None:
    """Print statistics of cpg.
    :param cpg_dict: cpg dict.
    """
    statistics = cpg_statistics(cpg_dict)
    logger.info('Statistics of cpg:')
    logger.info(f'Number of files: {len(statistics)}')
    nodes = list()
    edges = list()
    ast_edges = list()
    cfg_edges = list()
    dfg_edges = list()
    for _, value in statistics.items():
        nodes.append(value['num_of_nodes'])
        edges.append(value['num_of_edges'])
        ast_edges.append(value['num_of_ast_edges'])
        cfg_edges.append(value['num_of_cfg_edges'])
        dfg_edges.append(value['num_of_dfg_edges'])

    logger.info(f'Max number of nodes: {max(nodes)}')
    logger.info(f'Min number of nodes: {min(nodes)}')
    logger.info(f'Avg number of nodes: {sum(nodes) / len(nodes)}')
    logger.info(f'Median number of nodes: {np.median(nodes)}')
    logger.info(f'Max number of edges: {max(edges)}')
    logger.info(f'Min number of edges: {min(edges)}')
    logger.info(f'Avg number of edges: {sum(edges) / len(edges)}')
    logger.info(f'Median number of edges: {np.median(edges)}')

def cpg_statistics(cpg_dict: dict) -> dict:
    """Calculate statistics of cpg.
    :param cpg_dict: cpg dict.
    :return: statistics of cpg.
    {filename: {'num_of_funcs': num_of_funcs, 'num_of_nodes': num_of_nodes, 'num_of_edges': num_of_edges, 'num_of_ast_edges': num_of_ast_edges, 'num_of_cfg_edges': num_of_cfg_edges, 'num_of_dfg_edges': num_of_dfg_edges, 'num_of_cg_edges': num_of_cg_edges}}
    """
    statistics = dict()

    for _, value in cpg_dict.items():
        for func_cgnode in value:
            filename = func_cgnode.file_name
            if filename not in statistics:
                statistics[filename] = {'num_of_funcs': 0, 'num_of_nodes': 0, 'num_of_edges': 0, 'num_of_ast_edges': 0, 'num_of_cfg_edges': 0, 'num_of_dfg_edges': 0, 'num_of_cg_edges': 0}
            statistics[filename]['num_of_funcs'] += 1
            statistics[filename]['num_of_nodes'] += calculate_function_nodes(func_cgnode.cpg)
            edge_nums = calculate_function_edges(func_cgnode.cpg)
            statistics[filename]['num_of_ast_edges'] += edge_nums['ast']
            statistics[filename]['num_of_cfg_edges'] += edge_nums['cfg']
            statistics[filename]['num_of_dfg_edges'] += edge_nums['dfg']


            statistics[filename]['num_of_edges'] += statistics[filename]['num_of_ast_edges'] + statistics[filename]['num_of_cfg_edges'] + statistics[filename]['num_of_dfg_edges']
    
    return statistics

def calculate_function_nodes(cpg: DiGraph) -> int:
    """Calculate number of nodes in a function.
    :param cpg: cpg.
    :return: number of nodes in a function.
    """
    nodes = list(cpg.nodes)

    return len(nodes)

def calculate_function_edges(cpg: DiGraph) -> dict:
    """Calculate number of edges in a function.
    :param cpg: cpg.
    :return: number of edges in a function.
    {'ast': num_of_ast_edges, 'cfg': num_of_cfg_edges, 'dfg': num_of_dfg_edges, 'cg': num_of_cg_edges}
    """
    cpg_edges = list(cpg.edges)

    num_of_ast_edges = 0
    num_of_cfg_edges = 0
    num_of_dfg_edges = 0

    for _e in cpg_edges:
        start, end = _e
        edge_type = cpg[start][end]['edge_type']
        if edge_type == '100':
            num_of_ast_edges += 1
        elif edge_type == '010':
            num_of_cfg_edges += 1
        elif edge_type == '001':
            num_of_dfg_edges += 1
        elif edge_type == '110':
            num_of_ast_edges += 1
            num_of_cfg_edges += 1
        elif edge_type == '101':
            num_of_ast_edges += 1
            num_of_dfg_edges += 1
        elif edge_type == '011':
            num_of_cfg_edges += 1
            num_of_dfg_edges += 1
        elif edge_type == '111':
            num_of_ast_edges += 1
            num_of_cfg_edges += 1
            num_of_dfg_edges += 1
        else:
            print(f'{start} {end} {edge_type}: cannot be handled.')
            exit(-1)
    
    return {'ast': num_of_ast_edges, 'cfg': num_of_cfg_edges, 'dfg': num_of_dfg_edges}

def calculate_function_cg(callee, cpg_dict) -> int:
    """Calculate number of cg edges in a function.
    :param cpg: cpg.
    :return: number of cg edges in a function.
    """
    num_of_cg_edges = 0
    key = callee.get_key()
    if key not in cpg_dict:
        return num_of_cg_edges
    cg_nodes = cpg_dict[key]
    for cg_node in cg_nodes:
        entrynode = cg_node.entrynode
        num_of_cg_edges += 1
        fringe = cg_node.fringe
        num_of_cg_edges += len(fringe)
    
    return num_of_cg_edges
