class Stack():
    """Construct the stack structure using list
    """
    def __init__(self) -> None:
        self.__list = list()
    
    def is_empty(self):
        return self.__list == []
    
    def push(self, data):
        self.__list.append(data)
    
    def pop(self):
        if self.is_empty():
            return False
        return self.__list.pop()

class Queue():
    """Construct the queue structure using list
    """
    def __init__(self) -> None:
        self.__list = list()
    
    def is_empty(self):
        return self.__list == []
    
    def push(self, data):
        self.__list.append(data)
    
    def pop(self):
        if self.is_empty():
            return False
        
        return self.__list.pop(0)

class Edge():
    """Edge between two connected ASTNodes
    """
    def __init__(self, start: str, end: str, etype: str) -> None:
        self.start = start
        self.end = end
        self.etype = etype
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end
    
    def get_etype(self):
        return self.etype
