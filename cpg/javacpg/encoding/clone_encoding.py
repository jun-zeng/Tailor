from javacpg.encoding.encoding import encoding_bcb_clone

def encoding_clone(encode_path: str, func_dict: dict) -> None:
    """Encoding BCB functions for code clone detection.
    """
    file_dict = dict()
    all_cg_nodes = list()

    for _, v in func_dict.items():
        all_cg_nodes += v
    
    for cg_node in all_cg_nodes:
        file_id = cg_node.file_name
        cg_entities = list(cg_node.cpg.nodes)
        if file_id in file_dict:
            file_dict[file_id] += cg_entities
        else:
            file_dict[file_id] = cg_entities
    
    encoding_bcb_clone(encode_path, file_dict)