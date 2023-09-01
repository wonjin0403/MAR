from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

def set_dataloader(datasets: dict, batch_size: int, num_workers: int, shuffle: bool) -> dict:
    data_loader = {}
    for k, v in datasets.items():
        data_loader[k] = DataLoader(v, batch_size, shuffle=shuffle[k], num_workers=num_workers,
                                        drop_last=False, pin_memory=True)
    return data_loader
