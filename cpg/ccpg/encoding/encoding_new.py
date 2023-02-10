import os
from utils.setting import logger

def encoding_entity(clone_classification: str, encode_path: str, entities: list, start_idx: int = 0) -> bool:
    """Encoding entity from start index and store it to local file.
    
    Note: we hardcode the file name as `data/entity2id.txt`
    """

    file_name = 'entity2id.txt'
    file_name = os.path.join(encode_path, file_name)
    length = len(entities)
    entity_list = list()
    entity_id_dict = dict()

    for idx in range(length):
        match_statement = entities[idx][3]
        # if match_statement:
        if clone_classification == 'clone':
            if match_statement:
                _entity = list(entities[idx])[1:3] + [entities[idx][0]]
                entity_list.append(_entity)
            else:
                key = entities[idx][1] + entities[idx][2]
                if key not in entity_id_dict:
                    entity_id_dict[key] = list(entities[idx])[1:3] + [entities[idx][0]]
                else:
                    entity_id_dict[key].append(entities[idx][0])
        elif clone_classification == 'classification':
            if match_statement or len(entities[idx][2]) == 0:
                _entity = list(entities[idx])[1:3] + [entities[idx][0]]
                entity_list.append(_entity)
            else:
                key = entities[idx][1] + entities[idx][2]
                if key not in entity_id_dict:
                    entity_id_dict[key] = list(entities[idx])[1:3] + [entities[idx][0]]
                else:
                    entity_id_dict[key].append(entities[idx][0])
        else:
            logger.error('Unknown clone classification: {}' .format(clone_classification))
            exit(1)
        # if match_statement or len(entities[idx][2]) == 0:
        #     _entity = list(entities[idx])[1:3] + [entities[idx][0]]
        #     entity_list.append(_entity)
        # else:
        #     key = entities[idx][1] + entities[idx][2]
        #     if key not in entity_id_dict:
        #         entity_id_dict[key] = list(entities[idx])[1:3] + [entities[idx][0]]
        #     else:
        #         entity_id_dict[key].append(entities[idx][0])
    for _, _v in entity_id_dict.items():
        entity_list.append(_v)
    
    logger.info('Encoding entities: {}' .format(len(entity_list)))
    # write result to entity2id.txt
    with open(file_name, 'w') as entity2id:
        line_num = str(len(entity_list)) + '\n'
        entity2id.write(line_num)

        for idx in range(len(entity_list)):
            id = idx + start_idx
            _entity_item = [x.strip('\n').replace(',', ' ') for x in entity_list[idx]]
            line = str(id) + ',' + ','.join(_entity_item) + '\n'
            entity2id.write(line)
    
    return True

def load_entity_id(encode_path: str) -> dict:
    """Load entity pair dict, key is the hashkey and value is the id.
    """
    file_name = 'entity2id.txt'
    file_name = os.path.join(encode_path, file_name)
    
    if not os.path.exists(file_name):
        logger.error('Cannot find entity2id file, please check.')
        exit(-1)
    
    entity_id = dict()

    with open(file_name, 'r') as entity2id:
        lines = entity2id.readlines()
    
    for line in lines[1:]:
        line = line.strip('\n').split(',', 3)
        for hash_id in line[3].split(','):
            entity_id[hash_id] = line[0]
    
    # logger.info('Load Unique Entities: {}' .format(len(entity_id)))

    return entity_id

def load_rel_id(encode_path: str) -> dict:
    """Load relation id dict
    """
    file_name = 'rel2id.txt'
    file_name = os.path.join(encode_path, file_name)
    if not os.path.exists(file_name):
        logger.error('Cannot find rel2id file, please check it')
        exit(-1)
    
    rel_id = dict()
    with open(file_name, 'r') as rel2id:
        lines = rel2id.readlines()
    
    for line in lines[1:]:
        line = line.strip('\n').split(',')
        key, id = line[0], line[1]
        rel_id[key] = id
    
    if len(rel_id) != len(lines) - 1:
        logger.error('Exist Hash Collision, exit')
        exit(-1)
    logger.info(f'Load Relations: {len(rel_id)}')

    return rel_id

def encoding_triplet(encode_path: str, edges: list) -> bool:
    """Encode relation between CPGNode and store it to local file.
    """
    triplet_name = 'triple2id.txt'
    triplet_name = os.path.join(encode_path, triplet_name)

    entity_id = load_entity_id(encode_path)
    rel_id = load_rel_id(encode_path)

    edge_list = list()

    for edge in edges:
        start, end, edge_type = edge
        if start not in entity_id.keys() or end not in entity_id.keys():
            logger.error('Appear isolated edge (cannot find start or end node in graph).')
            exit(-1)
        s = entity_id[start]
        e = entity_id[end]
        e_t = rel_id[edge_type]
        
        edge_list.append([s, e, e_t])
    
    with open(triplet_name, 'w') as triplet2id:
        line_num = str(len(edge_list)) + '\n'
        triplet2id.write(line_num)
        for edge in edge_list:
            line = edge[0] + ',' + edge[1] + ',' + edge[2] + '\n'
            triplet2id.write(line)
    # logger.info('Encode triples {}' .format(len(edge_list)))

    return True

def encoding_statnodes(encode_path: str, s_es: list) -> bool:
    """Encode statement node entities.
    """
    statnodes_name = 'stat2entity.txt'
    statnodes_name = os.path.join(encode_path, statnodes_name)
    entity_id = load_entity_id(encode_path)

    stat_nodes = list()
    for stat_node in s_es:
        stat_id = list()
        for node in stat_node:
            if node not in entity_id.keys():
                logger.error('Appear isolated node (not appear in graph)')
                exit(-1)
            node_id = entity_id[node]
            stat_id.append(node_id)
        stat_nodes.append(stat_id)
    
    with open(statnodes_name, 'w') as stat2entity:
        line_num = str(len(stat_nodes)) + '\n'
        stat2entity.write(line_num)
        for nodes in stat_nodes:
            line = ','.join(nodes)
            line = line + '\n'
            stat2entity.write(line)
    
    logger.info('Encode statement nodes {}' .format(len(stat_nodes)))

    return True

def encoding_typetokens(encode_path: str, type_list: list, start_idx: int=0) -> bool:
    """Encode type tokens.
    """
    typetoken2id_name = 'typetoken2id.txt'
    typetoken2id_name = os.path.join(encode_path, typetoken2id_name)
    type_id = list()
    type_list = [x.strip('\n').replace(',', ' ') for x in type_list]
    type_list = list(set(type_list))
    length = len(type_list)

    for idx in range(length):
        id = idx + start_idx
        _type = type_list[idx]
        type_id.append([id, _type])
    
    with open(typetoken2id_name, 'w') as typetoken2id:
        line_num = str(length) + '\n'
        typetoken2id.write(line_num)
        for idx in range(length):
            line = str(type_id[idx][0]) + ',' + str(type_id[idx][1]) + '\n'
            typetoken2id.write(line)
    
    logger.info('Encode typetokens: {}' .format(length))

    return True

def load_typetoken_id(encode_path: str) -> dict:
    """Load type id pair dict, key is the typetoken and value is the id.
    """

    typetoken2id_name = 'typetoken2id.txt'
    typetoken2id_name = os.path.join(encode_path, typetoken2id_name)
    typetoken_id = dict()

    with open(typetoken2id_name, 'r') as typetoken2id:
        lines = typetoken2id.readlines()
    
    for line in lines[1:]:
        line = line.strip('\n').split(',', 1)
        key, id = line[1], line[0]
        if key in typetoken_id:
            logger.error('Duplicate typetoken {}' .format(key))

        typetoken_id[key] = id
    
    if len(typetoken_id) != len(lines) - 1:
        logger.error('Typetoken exists hash collision, exit.')
        exit(-1)
    
    return typetoken_id

def encoding_sequence(encode_path: str, func_list: list) -> bool:
    """Encode sequence
    """
    typetoken_id = load_typetoken_id(encode_path)
    typetoken_seq_file = 'typetoken_seq.txt'
    typetoken_seq_file = os.path.join(encode_path, typetoken_seq_file)
    seqs_list = list()

    for seq in func_list:
        seq_id = list()
        for _tt in seq:
            _tt = _tt.strip('\n').replace(',', ' ')
            if _tt not in typetoken_id.keys():
                logger.error('Cannot find type-id pair, exit')
                exit(-1)
            _tt_id = typetoken_id[_tt]
            seq_id.append(_tt_id)
        seqs_list.append(seq_id)
    
    with open(typetoken_seq_file, 'w') as seq2id:
        line_num = str(len(seqs_list)) + '\n'
        seq2id.write(line_num)
        for _seq in seqs_list:
            line = ','.join(_seq)
            line = line + '\n'
            seq2id.write(line)
    
    logger.info('Encode sequence: {}' .format(len(seqs_list)))

    return True

def encoding_entity2typetoken(encode_path: str) -> bool:
    """Encode entity 2 type/token (entity_id, type_id, token_id)
    """
    typetoken_id = load_typetoken_id(encode_path)
    entity2id_file = 'entity2id.txt'
    entity2typetoken_file = 'entity2typetoken.txt'
    entity2id_file = os.path.join(encode_path, entity2id_file)
    entity2typetoken_file = os.path.join(encode_path, entity2typetoken_file)
    entity_typetoken = list()
    
    with open(entity2id_file, 'r') as entity2id:
        lines = entity2id.readlines()
    
    for line in lines[1:]:
        _c1 = line.strip('\n').split(',', 1)
        entity_id = _c1[0]
        _c2 = _c1[1].split(',', 1)
        entity_type = _c2[0]
        entity_token = _c2[1].split(',', 1)[0]
        entity_type = entity_type.strip('\n')
        entity_token = entity_token.strip('\n')
        if entity_type not in typetoken_id.keys():
            logger.error('Cannot find type in typetoken dict.')
            exit(-1)
        type_id = typetoken_id[entity_type]
        if entity_token == '':
            token_id = -1
        elif entity_token not in typetoken_id.keys():
            logger.error(typetoken_id.keys())
            logger.error(entity_token)
            logger.error('Cannot find token in typetoken dict')
            exit(-1)
        else:
            token_id = typetoken_id[entity_token]
        entity_typetoken.append([entity_id, type_id, token_id])
    
    with open(entity2typetoken_file, 'w') as entity_tt:
        line_num = str(len(entity_typetoken)) + '\n'
        entity_tt.write(line_num)

        for _ett in entity_typetoken:
            _ett = [str(x) for x in _ett]
            line = ','.join(_ett)
            line = line + '\n'
            entity_tt.write(line)
    
    logger.info('Encode entity type token: {}' .format(len(entity_typetoken)))

    return True

def encoding_cg(encode_path: str, all_callees: list, func_dict: dict) -> bool:
    """Encode cg.
    """
    cg_name = 'cg2id.txt'
    cg_name = os.path.join(encode_path, cg_name)
    entity_id = load_entity_id(encode_path)

    all_edges = list()

    for callee in all_callees:
        key = callee.get_key()
        if key not in func_dict:
            continue
        start = callee.node_key
        if start not in entity_id:
            logger.error('Cannot find caller node')
            exit(-1)
        start_id = entity_id[start]
        
        cg_nodes = func_dict[key]
        for cg_node in cg_nodes:
            entrynode = cg_node.entrynode
            fringe = cg_node.fringe
            if entrynode not in entity_id:
                logger.error('Cannot find entry node')
                exit(-1)
            entrynode_id = entity_id[entrynode]
            _f_ids = list()
            for _f in fringe:
                if _f not in entity_id:
                    logger.error('Cannot find fringe')
                    exit(-1)
                _f_ids.append(entity_id[_f])
            
            all_edges.append([start_id, entrynode_id, '2'])
            for id in _f_ids:
                all_edges.append([id, start_id, '2'])
    
    logger.info('CG: {}' .format(len(all_edges)))
    with open(cg_name, 'w') as cg2id:
        line_num = str(len(all_edges)) + '\n'
        cg2id.write(line_num)
        
        for edge in all_edges:
            line = ','.join(edge)
            line = line + '\n'
            cg2id.write(line)
        
    
    return True

def merge_edges(encode_path: str) -> None:
    """Merge call graph with cpg
    """
    triple_file = 'triple2id.txt'
    cg_file = 'cg2id.txt'

    triple_file = os.path.join(encode_path, triple_file)
    cg_file = os.path.join(encode_path, cg_file)

    with open(triple_file, 'r') as tf:
        tf_lines = tf.readlines()
    
    with open(cg_file, 'r') as cf:
        cf_lines = cf.readlines()
    
    # logger.info('Ori_line: {}\t CG_line: {}' .format(len(tf_lines)-1, len(cf_lines)-1))
    line_num = int(tf_lines[0].strip('\n')) + int(cf_lines[0].strip('\n'))
    # logger.info('New lines: {}' .format(line_num))

    with open(triple_file, 'w') as tf:
        tf.write(str(line_num) + '\n')
        for line in tf_lines[1:] + cf_lines[1:]:
            tf.write(line)

def reduce_edges(encode_path: str) -> None:
    """Reduce repeated edges
    """
    triple_file = 'triple2id.txt'
    triple_file = os.path.join(encode_path, triple_file)
    with open(triple_file, 'r') as tf:
        tf_lines = tf.readlines()
    
    # logger.info(f'Before reduction repeated: {len(tf_lines)-1}')
    reduced_edges = list()
    for line in tf_lines[1:]:
        line = line.strip('\n')
        reduced_edges.append(line)
    
    reduced_edges = list(set(reduced_edges))
    
    with open(triple_file, 'w') as tf:
        line_num = str(len(reduced_edges)) + '\n'
        tf.write(line_num)
        for line in reduced_edges:
            line = line + '\n'
            tf.write(line)
    
    ast = 0
    cfg = 0
    dfg = 0

    for line in reduced_edges:
        line = line.split(',')
        if line[2] == '4':
            ast += 1
        elif line[2] == '2':
            cfg += 1
        elif line[2] == '1':
            dfg += 1
        elif line[2] == '6':
            ast += 1
            cfg += 1
        elif line[2] == '5':
            ast += 1
            dfg += 1
        elif line[2] == '3':
            cfg += 1
            dfg += 1
        elif line[2] == '7':
            ast += 1
            cfg += 1
            dfg += 1
        else:
            logger.error('Cannot identify edge')
            exit(-1)
    
    logger.info(f'Triples: {len(reduced_edges)}')
    logger.info(f'AST: {ast} \t CFG: {cfg} \t DFG: {dfg}')


def encoding_clone_entities(encode_path: str, functionality_dict: dict) -> None:
    """Encode functionality for code clone detection.
    """
    entity_id = load_entity_id(encode_path)
    clone_name = 'code_clone.txt'
    clone_name = os.path.join(encode_path, clone_name)
    
    with open(clone_name, 'w') as cf:
        line_num = str(len(functionality_dict)) + '\n'
        cf.write(line_num)
        for key, value in functionality_dict.items():
            f_id = (int(key) - 1) // 500 + 1
            entities = list()
            for entity in value:
                if entity not in entity_id:
                    logger.error('Cannot find functionality entity id')
                    exit(-1)
                entities.append(entity_id[entity])
            _es = ','.join(entities)
            line = str(f_id) + ',' + _es + '\n'
            cf.write(line)
    
    logger.info('Encode clone functionalities: {}' .format(len(functionality_dict)))
