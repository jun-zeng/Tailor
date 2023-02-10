from ccpg.encoding.encoding_new import *
from ccpg.encoding.query import *
from ccpg.encoding.clone_encoding import encoding_clone
from utils.setting import logger

# generating `typetoken2id.txt`
def batch_encoding_typetokens(encode_path: str, func_list: list, start_idx: int=0) -> None:
    """Encode typetokens of all functions. 

    Attributes:
        func_list -- A list including function(FunUnit instance).
        start_idx -- An interger of start index.
    """
    typetokens = list()
    for func in func_list:
        typetokens += list(set(func.gen_typetoken_sequence()))
    
    encoding_typetokens(encode_path, typetokens, start_idx)

# generating `typetoken_seq.txt`
def batch_encoding_sequences(encode_path: str, func_list: list) -> None:
    """Encode as a sequence of function. 
    """
    seqs = list()
    for func in func_list:
        seqs.append(func.gen_typetoken_sequence())
    
    encoding_sequence(encode_path, seqs)

# generating `entity2id.txt`
def batch_encoding_entities(clone_classification: str, encode_path: str, func_dict: dict, start_idx: int=0) -> None:
    """Encode entities.

    Attributes:
        func_dict: call graph dict.
    """
    all_entities = list()

    for key, value in func_dict.items():
        for func_cgnode in value:
            entities = entity_query(func_cgnode.cpg)
            all_entities += entities
    
    encoding_entity(clone_classification, encode_path, all_entities, start_idx)

# generating `entity2typetoken.txt`
def batch_encoding_entity2typetoken(encode_path: str) -> None:
    """Call encoding_entity2typetoken directly is ok
    """
    encoding_entity2typetoken(encode_path)

# generating `stat2entity.txt`
def batch_encoding_statnodes(encode_path: str, func_dict: dict) -> None:
    """Batch encoding statnodes.
    """
    all_statement_entities = list()

    for _, value in func_dict.items():
        for func_cgnode in value:
            statement_entities = stat_entities(func_cgnode.cpg)
            all_statement_entities += statement_entities

    encoding_statnodes(encode_path, all_statement_entities)

# generating `triple2id.txt`
def batch_encoding_triplet(encode_path: str, func_dict: dict) -> None:
    """Batch encoding edges
    """
    all_edges = list()

    for _, value in func_dict.items():
        for func_cgnode in value:
            edges = edge_query(func_cgnode.cpg)
            all_edges += edges
    
    encoding_triplet(encode_path, all_edges)

def batch_encoding_cg(encode_path: str, func_dict: dict) -> None:
    """Encoding call graph edge, regard all cg edges as control flow edge, edge type 2
    """
    all_callees = list()
    for _, v in func_dict.items():
        for func_cgnode in v:
            all_callees.append(func_cgnode.callees)

    all_callee_nodes = list()
    for callees in all_callees:
        for _, v in callees.items():
            all_callee_nodes += v


    encoding_cg(encode_path, all_callee_nodes, func_dict)
    merge_edges(encode_path)
    reduce_edges(encode_path)

def construt_rel_dict(encode_path: str) -> None:
    file_name = os.path.join(encode_path, 'rel2id.txt')
    if not os.path.exists(file_name):
        with open(file_name, 'w') as fn:
            fn.write('8\n')
            fn.write('100,4\n')
            fn.write('010,2\n')
            fn.write('001,1\n')
            fn.write('110,6\n')
            fn.write('101,5\n')
            fn.write('011,3\n')
            fn.write('111,7\n')
            fn.write('000,0')

def batch_encoding(clone_classification: str, encode_path: str, func_list: list, func_dict: dict) -> None:
    if not os.path.exists(encode_path):
        os.makedirs(encode_path)
    construt_rel_dict(encode_path)
    logger.info('Start encoding...')
    batch_encoding_typetokens(encode_path, func_list)
    batch_encoding_sequences(encode_path, func_list)
    batch_encoding_entities(clone_classification, encode_path, func_dict)
    batch_encoding_entity2typetoken(encode_path)
    batch_encoding_statnodes(encode_path, func_dict)
    batch_encoding_triplet(encode_path, func_dict)
    batch_encoding_cg(encode_path, func_dict)
    encoding_clone(encode_path, func_dict)
    logger.info(f'Encoding finished, check result in {encode_path}.')
