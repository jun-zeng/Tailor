from utils.setting import logger

class ASTNode():
    """Simplified ASTNode derived from tree-sitter Node
    """

    def __init__(self, node_key : str = None, node_type : str = None, node_token : str = "", start_idx : int = None, end_idx: int = None) -> None:
        if node_key == None or node_token == None or start_idx == None or end_idx == None:
            logger.debug('ASTNode lacks essential params.')
            exit(-1)
        self.node_key = node_key
        self.node_type = node_type
        self.node_token = node_token
        self.start_idx = start_idx
        self.end_idx = end_idx
    
    def get_ast_key(self) -> str:
        return self.node_key
        
    def get_ast_type(self) -> str:
        return self.node_type

    def get_ast_token(self) -> str:
        return self.node_token
