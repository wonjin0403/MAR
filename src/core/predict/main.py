import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys
import random
import numpy as np
import os
sys.path.append(os.path.realpath("./src"))
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything()

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    cfg["Trainer"] = instantiate(cfg["Trainer"])
    cfg["module"]["model"] = instantiate(cfg["module"]["model"])
    cfg["module"]["dataset"] = instantiate(cfg["module"]["dataset"])
    cfg["module"]["criterion"] = instantiate(cfg["module"]["criterion"])
    cfg["module"]["optimizer"]["params"] = cfg["module"]["model"].parameters()
    cfg["module"]["optimizer"] = instantiate(cfg["module"]["optimizer"])

    cfg["module"] = instantiate(cfg["module"])
    
    # test
    cfg["Trainer"].predict(cfg["module"], cfg["module"].predict_dataloader())
    
if __name__ == "__main__":
    main()
