from networkx import DiGraph
from utils.setting import logger

class CGNode():
    """Function Meta Structure for call graph construction.

    Consider the OJ dataset structure, we construct call graph with file as basic unit.

    Attributes:
        file_name: A string indicating the file name including current function.
        func_name: A string indicating the name of current function.
        parameter_type: A list including parameter types of current function.
        entrynode: A string indicating the entry point of cpg.
        fringe: A list including control flow graph out points.
        cpg: A DiGraph indicating code property graph of current function.
        callees: A dict including function call points in current function.
    """
    def __init__(self, file_name: str, func_name: str, parameter_type: list, entrynode: str, fringe: list, cpg: DiGraph, callees: dict) -> None:
        """Init CGNode class with specifc parameters.
        """
        self.file_name = file_name
        self.func_name = func_name
        self.parameter_type = parameter_type
        self.entrynode = entrynode
        self.fringe = fringe
        self.cpg = cpg
        self.callees = callees
    
    def get_key(self) -> str:
        """Generate a unique key for this function.
        """
        return self.file_name + '-' + self.func_name

class CalleeNode():
    """Class structure for callee function meta information.

    Attributes:
        node_key: A string indicating statement node key
        file_name: A string indicating file name including the callee function.
        callee_name: A string indicating the name of callee function.
        param_num: An integer indicating the number of arguments.
    """
    def __init__(self, node_key: str, file_name: str, callee_name: str, param_num: int) -> None:
        """Init CalleeNode class with specific parameters
        """
        self.node_key = node_key
        self.file_name = file_name
        self.callee_name = callee_name
        self.param_num = param_num

    def get_key(self) -> str:
        """Generate a unique key for this callee function
        
        As for the dataset (OJ), we consider call graph in one single file.
        For C language, it is enough to use the combination of file name and function name as key.
        """
        return self.file_name + '-' + self.callee_name
    
    def print_callee(self) -> None:
        """Print callee function information for debugging.
        """
        logger.warn('File Name:\t{}\tCallee Name:\t{}' .format(self.file_name, self.callee_name))