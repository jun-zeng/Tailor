from networkx import DiGraph
from utils.setting import logger

class CGNode():
    """Function Meta Structure for call graph construction.

    attributes:
        file_name -- name of the file includes current function.
        import_header -- import headers of the class.
        func_name -- name of current function.
        parameter_type -- argument types.
        parameter_name -- argument names.
        entrynode -- control flow graph entry point.
        fringe -- control flow graph out point.
        cpg -- code property graph of each function.
    """
    def __init__(self, file_name: str, import_header: list, func_name: str, parameter_type: list, parameter_name: list, entrynode: str, fringe: list, cpg: DiGraph, callees: dict) -> None:
        if entrynode == None or fringe == [] or cpg == None:
            logger.error('Constructing CGNode failed, lacks entry node, fringe or cpg, exit')
            exit(-1)
        self.file_name = file_name
        self.import_header = import_header
        self.func_name = func_name
        self.parameter_type = parameter_type
        self.parameter_name = parameter_name
        self.entrynode = entrynode
        self.fringe = fringe
        self.cpg = cpg
        self.callees = callees
    
    # TODO
    def is_matched_callee(self) -> bool:
        """Match current function is matched with the caller or not.
        """
        pass


class CalleeNode():
    """Class structure for callee function meta information.
    
    attributes:
        node_key -- statement key
        callee_name -- name of the callee function.
        param_num -- number of arguments
        param_type -- types inferred for arguments.
    """
    def __init__(self, node_key: str, callee_name: str, param_num: int, param_type: list) -> None:
        self.node_key = node_key
        self.callee_name = callee_name
        self.param_num = param_num
        self.param_type = param_type
    
    def get_key(self) -> str:
        """Generate a key for this callee function.
        """
        return self.callee_name + '-' + str(self.param_num)
    
    def print_callee(self) -> None:
        print('Callee Name:\t{}\nArgs Num:\t{}\nArgs Type:\t{}' .format(self.callee_name, self.param_num, ','.join(self.param_type)))
