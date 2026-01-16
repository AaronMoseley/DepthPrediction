class ArgumentManager:
    def __init__(self, argv:list[str]) -> None:
        self.fileName = argv[0]
        
        for i in range(1, len(argv)):
            if not argv[i].startswith("--"):
                continue

            equalPos = argv[i].find("=")
            variableName = argv[i][2:equalPos]
            value = argv[i][equalPos + 1:]

            self.__dict__[variableName] = value

    def __getitem__(self, name:str) -> str|None:        
        if name not in self.__dict__:
            return None
        
        return self.__dict__[name]
    
    def GetFileName(self) -> str:
        return self.fileName