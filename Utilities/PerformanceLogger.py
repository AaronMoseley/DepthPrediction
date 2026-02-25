import torch

class PerformanceLogger():
    def __init__(self) -> None:
        return None

    def LogData(self, data:dict, step:int=None) -> None:
        return None
    
    def LogImage(self, tensorList:list[torch.Tensor], epochIndex:int, batchIndex:int) -> None:
        return None
    
    def FinishRun(self) -> None:
        return None