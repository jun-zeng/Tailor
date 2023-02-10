import uuid
import tree_sitter

from treelib import Tree
from utils.data_structure import Queue
from utils.setting import logger
from javacpg.sast.ast_node import ASTNode
from javacpg.sast.query_pattern import JAVA_QUERY

def generate_ast_key(file_name: str, func_name: str, parsed_node: tree_sitter.Node) -> str:
    """ Generate unique key value for each ASTNode

    attributes:
        file_name -- name of the file including current node\\
        parsed_node -- instance of tree_sitter Node is parsed now
    
    returns:
        key_value -- unique key string for parsed_node
    """
    # construct unique string for each tree_sitter Node
    key_str = file_name + '-' + func_name + '-' + parsed_node.type + '-' + str(parsed_node.start_byte) + '-' + str(parsed_node.end_byte)
    key_value = uuid.uuid3(uuid.NAMESPACE_DNS, key_str)
    key_value = str(key_value).replace('-', '')

    return key_value

def build_func_sast(file_name: str, func_name: str, func_tree : tree_sitter.Node, src_code : bytes, exclude_type: list) -> Tree:
    """Build simplified AST (sast) with function as the basic unit

    attributes:
        file_name -- name of the file including current function\\
        func_tree -- function ast generated by tree-sitter\\
        src_code -- serial source code for token querying\\
        exclude_type -- identifier types ignored
    
    returns:
        s_ast -- simplified ast organized by ASTNode
    """
    s_ast = Tree()
    
    # create root node for this function
    root_node = func_tree
    root_key = generate_ast_key(file_name, func_name, root_node)
    has_child = len(root_node.children)
    if not has_child:
        root_token = src_code[root_node.start_byte:root_node.end_byte]
    else:
        root_token = ''
    root_ast = ASTNode(root_key, root_node.type, root_token, root_node.start_byte, root_node.end_byte)
    s_ast.create_node(tag=root_node.type, identifier=root_key, data=root_ast)

    query = JAVA_QUERY()
    ret_node = query.method_ret_query().captures(root_node)[0][0]
    
    # create ret node for each ast
    ret_key = generate_ast_key(file_name, func_name, ret_node)
    ret_token = src_code[ret_node.start_byte:ret_node.end_byte].decode('utf8')
    ret_astnode = ASTNode(ret_key, 'ret_type', ret_token, ret_node.start_byte, ret_node.end_byte)

    queue = Queue()
    queue.push(root_node)
    while not queue.is_empty():
        current_node = queue.pop()

        for child in current_node.children:
            child_type = str(child.type)
            if child_type in exclude_type:
                logger.debug('Ignore node type {}' .format(child_type))
                continue
            child_key = generate_ast_key(file_name, func_name, child)
            child_token = ''
            has_child = len(child.children) > 0
            if not has_child:
                child_token = src_code[child.start_byte:child.end_byte].decode('utf8')
            parent_identifier = generate_ast_key(file_name, func_name, current_node)
            s_ast.create_node(tag=child_type, identifier=child_key, parent=parent_identifier, data=ASTNode(child_key, child_type, child_token, child.start_byte, child.end_byte))

            queue.push(child)
    if s_ast.get_node(ret_key) == None:
        logger.debug('Creating return astnode fails. Exit.')
        exit(-1)
    
    # remove return parameter node, and place it as return node.
    s_ast.remove_node(ret_key)
    s_ast.create_node(tag=ret_node.type, identifier=ret_key, parent=root_key, data=ret_astnode)

    return s_ast
