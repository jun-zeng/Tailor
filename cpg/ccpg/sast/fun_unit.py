from treelib import Tree
from utils.data_structure import Stack
from utils.setting import logger

class FunUnit():
    """Maintain data for each function (e.g., file_name, func_name, parameter_type)
    """
    def __init__(self, sast: Tree, file_name: str = None, func_name: str = None, parameter_type: list = [], include_path: list = []) -> None:
        """Constructor of FunUnit class
        """
        if file_name == None or func_name == None:
            logger.debug('FunUnit lacks essential params.')
            exit(-1)
        self.sast = sast
        self.file_name = file_name
        self.func_name = func_name
        self.parameter_type = parameter_type
        self.include_path = include_path
    
    def gen_type_sequence(self) -> list:
        """Depth-first search for generating type sequence.
        """
        sequence = list()
        root = self.sast.root

        stack = Stack()
        stack.push(root)

        while not stack.is_empty():
            current_node = stack.pop()
            _node_data = self.sast.get_node(current_node).data
            current_node_type = _node_data.node_type
            sequence.append(current_node_type)
            children = self.sast.children(current_node)
            children.reverse()
            for child in children:
                stack.push(child.identifier)
        
        return sequence
    
    def has_type(self, type: str) -> bool:
        """Determine whether the sast contains specific type.
        """
        type_sequence = self.gen_type_sequence()
        if type in type_sequence:
            return True
        
        return False
    
    def gen_typetoken_sequence(self) -> list:
        """Depth-first search for generating type sequence.
        """
        sequence = list()
        root = self.sast.root

        stack = Stack()
        stack.push(root)

        while not stack.is_empty():
            current_node = stack.pop()
            _node_data = self.sast.get_node(current_node).data
            current_node_type = _node_data.node_type
            current_node_token = _node_data.node_token
            sequence.append(current_node_type)
            if current_node_token:
                sequence.append(current_node_token)
            children = self.sast.children(current_node)
            children.reverse()
            for child in children:
                stack.push(child.identifier)
            
        
        return sequence