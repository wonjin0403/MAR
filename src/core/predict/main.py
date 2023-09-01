import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
import random
import numpy as np
import os
sys.path.append(os.path.realpath(“./src”))
import core
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ[“PYTHONHASHSEED”] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
seed_everything()

def infer(model, dataset, dataloader):
    for idx, batch in enumerate(dataloader):
        output = model(batch)

@hydra.main(version_base=None, config_path=“../../config”, config_name=“config”)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    model = instantiate(cfg[“module”][“model”])
    dataset = instantiate(cfg[“module”][“dataset”])
    dataloader = torch.data.Dataloader(dataset, batch_size=cfg[“batch_size”], num_workers=cfg[“num_workers”])
    infer(model, dataset, dataloader)

if __name__ == “__main__“:
    main()