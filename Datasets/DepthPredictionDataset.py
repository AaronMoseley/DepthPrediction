from torch.utils.data import Dataset
import torch

class DepthPredictionDataset(Dataset):
    def __init__(self, datasetName:str, indices:list[int]=None) -> None:
        super().__init__()

        if indices is not None:
            self.indices = indices
            return
        
        self.indices = []

    def __len__(self) -> int:
        pass

    def __getitem__(self, index:int) -> torch.Tensor:
        pass