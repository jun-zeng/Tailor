import uuid

from treelib import Tree
from utils.data_structure import Edge, Queue, Stack
from utils.setting import logger


class FunUnit():
    """Maintain data for each function (e.g., file_name, func_name, parameter_num)

    attributes:
        sast -- instance of simplified Abstract Syntax Tree\\
        file_name -- path of the file including current function\\
        func_name -- name of current function\\
        parameter_type -- parameter type of current function\\
        parameter_name -- parameter name of current function \\
        import_header -- header used by this function\\
        field_params -- class field parameters for data dependency graph
        
    """
    
    def __init__(self, sast: Tree, file_name: str = None, func_name: str = None, parameter_type: list = [], parameter_name: list = [], import_header: list = [], field_params: list = []) -> None:
        """ Constructor of FunUnit class
        """
        if file_name == None or func_name == None:
            logger.debug('FunUnit lacks essential params. file_name: {}, func_name: {}' .format(file_name, func_name))
            exit(-1)
        self.sast = sast
        self.file_name = file_name
        self.func_name = func_name
        self.parameter_type = parameter_type
        self.parameter_name = parameter_name
        self.import_header = import_header
        self.field_params = field_params
    
    def format_sast(self) -> list:
        """Transform tree structure (Multi-way Tree) to graph structure (G=(V, E))

        attributes:
            self -- self class
        
        returns:
            [nodes, edges] -- list including ast nodes and corresponding edges, particularly, nodes is a node directory and edges is a edge list.
        """
        nodes = dict()
        edges = list()
        root = self.sast.root

        nodes[root] = self.sast.get_node(root).data

        queue = Queue()
        queue.push(root)

        while not queue.is_empty():
            current_node = queue.pop()

            for child in self.sast.children(current_node):
                child_identifier = child.identifier
                child_data = child.data
                nodes[child_identifier] = child_data
                edges.append(Edge(current_node, child_identifier, 'ast'))
                queue.push(child_identifier)
        
        return [nodes, edges]
    
    def gen_func_key(self) -> str:
        """Generate one identifier key for current function with file name, function name and parameter number

        attributes:
            self
        
        returns:
            func_key -- one string label the function.
        """
        label_str = self.file_name + self.func_name + str(len(self.parameter_type))
        func_key = uuid.uuid3(uuid.NAMESPACE_DNS, label_str)
        func_key = str(func_key).replace('-', '')

        return func_key

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
        """Determine whether the sast contains specifc type.
        """
        type_sequence = self.gen_type_sequence()
        if type in type_sequence:
            return True
        
        return False
            