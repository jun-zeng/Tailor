from javacpg.sast.ast_node import ASTNode
from utils.setting import logger

class CPGNode():
    """Node structure of Code Property Graph.
    """
    def __init__(self, ast_node: ASTNode = None) -> None:
        if ast_node == None:
            logger.debug('CPGNode initialization lacks ASTNode.')
            exit(-1)
        self.node_key = ast_node.node_key
        self.node_type = ast_node.node_type
        self.node_token = ast_node.node_token
        self.start_idx = ast_node.start_idx
        self.end_idx = ast_node.end_idx
        # whether CPGNode corresponds to a code statement or not
        self.match_statement = False
    
    def is_statement_node(self) -> bool:
        """Judge whether current CPGNode matches one code statement or not.

        attributes:
            self class
        
        returns:
            True/False -- if current CPGNode matches one statement, True, else False.
        """
        return self.match_statement
    
    def set_statement_node(self) -> bool:
        """Set match_statement as True if current CPGNode matches one statement.
        """
        self.match_statement = True

        return True

    def get_cpg_key(self) -> str:
        """Return the key of current CPGNode.
        """
        return self.node_key
    
    def get_cpg_type(self) -> str:
        """Return the type of current CPGNode.
        """
        return self.node_type
    
    def get_cpg_token(self) -> str:
        """Return the token of current CPGNode.
        """
        return self.node_token
    