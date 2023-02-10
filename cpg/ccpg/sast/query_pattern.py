from tree_sitter import Language
import os

class C_QUERY():
    """Provide query pattern for c language
    """
    def __init__(self, library_path: str = None) -> None:
        if library_path == None:
            path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            self.library = path + '/lib/my-languages.so'
        else:
            self.library = library_path
        
        self.C_LANGUAGE = Language(self.library, 'c')
    
    def include_path_query(self):
        """Include path matching pattern
        """
        query = self.C_LANGUAGE.query("""
        (translation_unit (preproc_include path: * @include_path))""")
        
        return query

    def function_definition_query(self):
        """Function definition matching pattern
        """
        query = self.C_LANGUAGE.query("""
        (translation_unit (function_definition) @function)""")
        
        return query
    
    def function_declarator_query(self):
        """Function declaration matching pattern
        """
        query = self.C_LANGUAGE.query("""
        (function_declarator declarator: (identifier) @function_name)""")
        
        return query
    
    def function_parameters_type_query(self):
        """Function parameter type matching pattern
        """
        query = self.C_LANGUAGE.query("""
        (function_declarator
        parameters: (
            parameter_list (
                parameter_declaration type: * @type)))""")
        
        return query
    
    def function_ret_query(self):
        """Function return type matching pattern
        """
        query = self.C_LANGUAGE.query("""
        (function_definition
        type: * @ret_type)""")
        
        return query