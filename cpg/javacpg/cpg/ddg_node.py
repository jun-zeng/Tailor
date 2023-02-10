from utils.setting import logger

class DDGNode():
    """Node structure used for Data Dependency Graph construction. In general, we traverse the ast-based control flow graph, find each control flow node's def-use information.
    """
    def __init__(self, node_key: str = None, node_type: str = None, defs: list = [], uses : list = [], unknow: list = []) -> None:
        if node_key == None:
            logger.error('DDGNode initialization need node type, exit.')
            exit(-1)
        self.node_key = node_key
        self.node_type = node_type
        self.defs = defs
        self.uses = uses
        self.unknown = unknow
    
    def get_key(self) -> str:
        return self.node_key
        
    def get_defs(self) -> list:
        return self.defs
    
    def get_uses(self) -> list:
        return self.uses
    
    def get_unknown(self) -> list:
        return self.unknown
    
    def print_defs_uses(self) -> None:
        print('***DDGNode info***')
        print('Node Key: {}\nNode Type: {}\nNode Defs: {}\nNode Uses: {}\nNode Unknown: {}' .format(self.node_key, self.node_type, ','.join(self.defs), ','.join(self.uses), ','.join(self.unknown)))
