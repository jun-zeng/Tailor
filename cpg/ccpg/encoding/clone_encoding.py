from ccpg.encoding.encoding_new import encoding_clone_entities

def encoding_clone(encode_path: str, func_dict: dict) -> None:
    """Encoding OJ functionalities [1-15] for code clone detection.
    """
    functionality_dict = dict()

    all_cg_nodes = list()

    for _, v in func_dict.items():
        all_cg_nodes += v
    
    for cg_node in all_cg_nodes:
        key = cg_node.file_name
        cg_entities = list(cg_node.cpg.nodes)
        if key in functionality_dict:
            functionality_dict[key] += cg_entities
        else:
            functionality_dict[key] = cg_entities
    
    encoding_clone_entities(encode_path, functionality_dict)

