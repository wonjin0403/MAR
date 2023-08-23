import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lightning as pl

import sys
sys.path.append("/app/Final_MAR_code/src")
from common.utils.set_dataloader import set_dataloader
from common.utils.utils import pearson_correlation_coeff, save_as_dicom

class MAR(pl.LightningModule):
    def __init__(self,
                 model: Module, 
                 criterion: Module, 
                 optimizer: Module,
                 save_output_only: bool, 
                 test_save_path: str,
                 dataset: dict,
                 batch_size: int, num_worker: int):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_output_only = save_output_only
        self.test_save_path = test_save_path
        self.dataloader = set_dataloader(dataset["train"], dataset["validation"], dataset["test"], batch_size, num_worker)
        self.validation_step_outputs = []
        self.testing_step_outputs = []

    def train_dataloader(self):
        return self.dataloader["train"]
    
    def val_dataloader(self):
        return self.dataloader["validation"]
    
    def test_dataloader(self):
        return self.dataloader["test"]
    
    def configure_optimizers(self):
        return self.optimizer
    
    def step(self, input_ct):
        output = self.model(input_ct)
        return output
    
    def _metric(self, target, output):
        _pcc = pearson_correlation_coeff(target, output)
        _ssim = compare_ssim(np.moveaxis(target, 0, -1), np.moveaxis(output, 0, -1), multichannel=True)
        _psnr = compare_psnr(target, output, data_range=2)
        _mse = (np.square(target - output)).mean(axis=None)
        return _pcc, _ssim, _psnr, _mse

    def training_step(self, batch, batch_idx: int) -> dict:
        input_, target_, _, _ = batch
        output_ = self.step(input_)
        loss = self.criterion(output_, target_)
        return {"loss":loss}
    
    def on_training_epoch_end(self, outputs: list) -> None:
        loss = [output["loss"] for output in outputs]
        self.log("train/loss", np.mean(loss), on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx: int) -> dict:
        input_, target_, _, _ = batch
        output_ = self.step(input_)
        loss = self.criterion(output_, target_)
        pcc_list, ssim_list, psnr_list, mse_list = [], [], [], []
        for idx in range(input_.shape[0]):
            _pcc, _ssim, _psnr, _mse = self._metric(target_[idx].cpu().numpy(), output_[idx].cpu().numpy())
            pcc_list.append(_pcc)
            ssim_list.append(_ssim)
            psnr_list.append(_psnr)
            mse_list.append(_mse)
        results = {"val_loss": loss, "pcc": pcc_list, "ssim": ssim_list, "psnr": psnr_list, "mse": mse_list}
        self.validation_step_outputs.append(results)
        return results
    
    def on_validation_epoch_end(self) -> None:
        loss_list, pcc_list, ssim_list, psnr_list, mse_list = [], [], [], [], []
        for output in self.validation_step_outputs:
            loss_list.append(output["val_loss"].item())
            pcc_list.extend(output["pcc"])
            ssim_list.extend(output["ssim"])
            psnr_list.extend(output["mse"])
        self.validation_step_outputs.clear()
        self.log("valid/loss", np.mean(loss_list), on_epoch=True, sync_dist=True)
        self.log("valid/pcc", np.mean(pcc_list), on_epoch=True, sync_dist=True)
        self.log("valid/ssim", np.mean(ssim_list), on_epoch=True, sync_dist=True)
        self.log("valid/psnr", np.mean(psnr_list), on_epoch=True, sync_dist=True)
        self.log("valid/mse", np.mean(mse_list), on_epoch=True, sync_dist=True)

    def testing_step(self, batch, batch_idx: int) -> dict:
        input_, target_, mask_, imgName = batch
        output_ = self.step(input_)
        loss = self.criterion(output_, target_)
        pcc_list, ssim_list, psnr_list, mse_list = [], [], [], []
        for idx in range(input_.shape[0]):
            _pcc, _ssim, _psnr, _mse = self._metric(target_[idx].cpu().numpy(), output_[idx].cpu().numpy())
            pcc_list.append(_pcc)
            ssim_list.append(_ssim)
            psnr_list.append(_psnr)
            mse_list.append(_mse)
            if self.save_output_only:
                save_as_dicom(output=output_[idx], test_save_path=self.test_save_path, imgName=imgName[idx])
            else:
                save_as_dicom(input=input_[idx], 
                              target_=target_[idx], 
                              output=output_[idx], 
                              test_save_path=self.test_save_path, 
                              imgName=imgName[idx])
        results = {"loss":loss, "pcc": pcc_list, "ssim": ssim_list, "psnr": psnr_list, "mse": mse_list, "imgName": imgName}
        self.testing_step_outputs.append(results)
        return results
    

    def on_testing_step_end(self) -> None:
        loss_list, pcc_list, ssim_list, psnr_list, mse_list = [], [], [], [], []
        for output in self.testing_step_outputs:
            loss_list.append(output["loss"].item())
            pcc_list.extend(output["pcc"])
            ssim_list.extend(output["ssim"])
            psnr_list.extend(output["mse"])
        print("------------------")
        print("Evaluation Result")
        print(f"pcc: {np.mean(pcc_list)}")
        print(f"ssim: {np.mean(ssim_list)}")
        print(f"psnr: {np.mean(psnr_list)}")
        print(f"psnr: {np.mean(mse_list)}")
