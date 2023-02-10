from tree_sitter import Language
import logging

logger = logging.getLogger('Tree-sitter-build')

def build_language_libs() -> None:
    """ Build language library for py-tree-sitter

    attributes:
        xxx
    
    returns:
        xxx
    """
    output_path = './libs/my-languages.so'
    # Include one or more languages
    repo_paths = [
        'langs/tree-sitter-java',
        'langs/tree-sitter-c',
        'langs/tree-sitter-cpp',
    ]
    res = Language.build_library(output_path, repo_paths)
    if not res:
        logger.error('Compiling language library fail.')
    
    
if __name__ == "__main__":
    build_language_libs()