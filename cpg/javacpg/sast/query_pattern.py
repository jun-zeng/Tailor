from tree_sitter import Language
import os

class JAVA_QUERY():
    """Provide query pattern for java language
    """
    def __init__(self, library_path: str = None) -> None:
        if library_path == None:
            path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            self.library = path + '/ts_build/libs/my-languages.so'
        else:
            self.library = library_path

        self.JV_LANGUAGE = Language(self.library, 'java')

    def class_method_query(self):
        """Class method matching pattern
        """
        query = self.JV_LANGUAGE.query("""
        (class_declaration
        body: (class_body 
            (method_declaration) @method))
        """)

        return query
    
    def method_declaration_query(self):
        """Method declaration matching pattern
        """
        query = self.JV_LANGUAGE.query("""
        (method_declaration
        name: (identifier) @method_name)
        """)

        return query
    
    def method_parameters_query(self):
        """Method parameters matching pattern
        """
        query = self.JV_LANGUAGE.query("""
        (method_declaration
        parameters: (
            formal_parameters
            (formal_parameter 
            type: * @type
            name: * @name)))
        """)

        return query
    
    def method_ret_query(self):
        """Method return parameter matching pattern
        """
        query = self.JV_LANGUAGE.query("""
        (method_declaration
        type: * @ret_type)
        """)

        return query
    
    def method_invocation_query(self):
        """Method invocation matching pattern
        """
        query = self.JV_LANGUAGE.query("""
        (method_invocation
            name: (identifier) @callee_name
            arguments: (argument_list * @types))
        """)

        return query

    def import_header_query(self):
        """Import header matching pattern
        """
        query = self.JV_LANGUAGE.query("""
        ((import_declaration) @import_header)
        """)

        return query
    
    def class_filed_query(self):
        """Class field matching pattern
        """
        query = self.JV_LANGUAGE.query("""
        (field_declaration
        declarator: (
            variable_declarator 
            name: * @class_field))
        """)
        return query