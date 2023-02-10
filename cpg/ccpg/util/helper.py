"""Provide some basic help functions.
"""
import os
import pickle

from ccpg.sast.fun_unit import FunUnit
from ccpg.cpg.ast_constructor import gen_ast_cpg
from ccpg.cpg.cfg_constructor import cfg_build
from ccpg.cpg.ddg_constructor import ddg_build
from ccpg.util.visualize import visualize_ast_cpg

def has_entry_function(func_list: list) -> bool:
    """Ensure each file having the entry function main.

    main entry function is necessary for c functionality, check it.
    """
    has_main = False
    for func in func_list:
        if func.func_name in ['main', 'Main', 'MAIN']:
            has_main = True
            break
    
    return has_main

def traverse_files(dir_path: str = None) -> list:
    all_files = list()
    walk_tree = os.walk(dir_path)
    for root, _, files in walk_tree:
        for file in files:
            all_files.append(os.path.join(root, file))
    
    return all_files

def check_extension(file_name: str, extension: str) -> bool:
    _extension = os.path.splitext(file_name)[-1][1:]
    if _extension == extension:
        return True
    return False

def traverse_src_files(dir_path: str, extension: str) -> list:
    """Obtain all source files we want to parse.

    attributes:
        dir_path -- the directory path we want to parse.
        extension -- the file extension we want to parse (e.g., 'java')
    
    returns:
        files -- list including files we want to parse.
    """
    files = list()
    all_files = traverse_files(dir_path)
    for file in all_files:
        if check_extension(file, extension):
            files.append(file)
    
    return files

def check_key_repeat(entities: list) -> bool:
    _exist_key = list()
    for entity in entities:
        node_key = entity[0]
        if node_key not in _exist_key:
            _exist_key.append(node_key)
        else:
            return True
    
    return False

def visualize_helper(func: FunUnit, file_name: str) -> bool:
    func_root = func.sast.root
    func_cpg = gen_ast_cpg(func.sast)
    _, _ = cfg_build(func_cpg, func_root)
    ddg_build(func_cpg, func_root)
    visualize_ast_cpg(func_cpg, file_name)

    return True

def load_inter_results(dir_path: str) -> list:
    """Load inter results of function list and function dict
    """
    func_list_path = os.path.join(dir_path, 'func_list.pkl')
    func_dict_path = os.path.join(dir_path, 'func_dict.pkl')

    if not os.path.exists(func_list_path) or not os.path.exists(func_dict_path):
        print('Cannot find stored inter results.')
        exit(-1)

    fl = open(func_list_path, 'rb')
    func_list = pickle.load(fl)
    fl.close()

    fd = open(func_dict_path, 'rb')
    func_dict = pickle.load(fd)
    fd.close()

    return [func_list, func_dict] 

def store_inter_results(dir_path: str, func_list: list, func_dict: dict) -> bool:
    """Store inter results of function list and function dict
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    func_list_path = os.path.join(dir_path, 'func_list.pkl')
    func_dict_path = os.path.join(dir_path, 'func_dict.pkl')

    try:
        fl = open(func_list_path, 'wb')
        pickle.dump(func_list, fl)
        fl.close()

        fd = open(func_dict_path, 'wb')
        pickle.dump(func_dict, fd)
        fd.close

        return True
    except:
        return False