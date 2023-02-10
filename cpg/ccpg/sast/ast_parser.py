import tree_sitter
from tree_sitter import Language, Parser
from utils.setting import logger
import os

class ASTParser():
    """Generate parser for c language
    """
    def __init__(self, lib_path: str = None) -> None:
        """Binding tree-sitter library for c langurage with python
        """
        self.lib_path = lib_path
        if self.lib_path == None:
            logger.debug('Cannot find Language Library, adapting `my-language.so` in lib as default')
            path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            self.lib_path = path + '/lib/my-languages.so'
        self.parser = Parser()
        self.parser.set_language(Language(self.lib_path, 'c'))
    
    def parse(self, src_code: bytes = None) -> tree_sitter.Tree:
        return self.parser.parse(src_code)