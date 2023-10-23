import torch
import numpy as np
import pandas as pd
from torch.nn import Module
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lightning as pl
import os
import sys
import json
sys.path.append(os.path.realpath("../../src/"))
from common.utils.set_dataloader import set_dataloader
from common.utils.utils import pearson_correlation_coeff, save_as_dicom_test, save_as_dicom_infer

class MAR(pl.LightningModule):
    def __init__(self,
                 model: Module, 
                 criterion: Module, 
                 optimizer: Module,
                 scheduler: Module,
                 save_output_only: bool, 
                 test_save_path: str,
                 dataset: dict,
                 batch_size: int, num_worker: int, save_path: str, shuffle: bool=False):
        super().__init__()
        self.model = model
        if save_path is not None:
            self.load_from_checkpoint(save_path)
        self.batch_size = batch_size
        self.save_path = save_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_output_only = save_output_only
        self.test_save_path = test_save_path
        self.dataloader = set_dataloader(dataset, batch_size, num_worker, shuffle)
        self.validation_step_outputs = []
        self.testing_step_outputs = []

    def train_dataloader(self):
        return self.dataloader["train"]
    
    def val_dataloader(self):
        return self.dataloader["validation"]
    
    def test_dataloader(self):
        return self.dataloader["test"]
    
    def predict_dataloader(self):
        return self.dataloader["predict"]
    
    def load_from_checkpoint(self, path: str) ->None:
        checkpoint = torch.load(path)
        new_dict = {}
        for key, v in checkpoint["state_dict"].items():
            new_dict[key.replace("model.", "")] = v
        print(self.model.load_state_dict(new_dict))
        print(f"load checkpoint from {path}")
    
    def configure_optimizers(self):
        return self.optimizer #{"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
    
    def lr_scheduler_step(self, scheduler, metric):
        self.scheduler.step(epoch=self.current_epoch) 
        
    def step(self, input_ct):
        output = self.model(input_ct)
        return output
    
    def _metric(self, target, output):
        _pcc = pearson_correlation_coeff(target[:,:,160:-180,180:-180], output[:,:,160:-180,180:-180])
        _ssim = compare_ssim(np.moveaxis(target[:,:,160:-180,180:-180], 0, -1), np.moveaxis(output[:,:,160:-180,180:-180], 0, -1), multichannel=True, channel_axis=2, data_range=float(2))
        _psnr = compare_psnr(target[:,:,160:-180,180:-180], output[:,:,160:-180,180:-180], data_range=2)
        _mse = (np.square(target[:,:,160:-180,180:-180] - output[:,:,160:-180,180:-180])).mean(axis=None)
        metrics = 100*_mse + (-1*_ssim)
        return _pcc, _ssim, _psnr, _mse, metrics

    def training_step(self, batch, batch_idx: int) -> dict:
        input_, target_, bone_mask_, tissue_mask_, _, _ = batch
        output_ = self.step(input_)
        loss = self.criterion(output_, target_, tissue_mask_)
        self.log("train/loss", loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        # self.log("train/loss_mse", loss_all, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        # self.log("train/loss_tissue_mse", loss_tissue, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train/learning rate', self.scheduler.get_last_lr()[0], on_epoch=True, batch_size=self.batch_size)
        return {"loss":loss}
    
    # def on_training_epoch_end(self, outputs: list) -> None:
    #     loss = [output["loss"] for output in outputs]
    #     self.log("train/loss", np.mean(loss), on_epoch=True, sync_dist=True)
    #     self.log('train/learning rate', self.scheduler.get_lr()[0], on_epoch=True)

    def validation_step(self, batch, batch_idx: int) -> dict:
        input_, target_, bone_mask_, tissue_mask_, _, _ = batch
        output_ = self.step(input_)
        loss = self.criterion(output_, target_, tissue_mask_)
        pcc_list, ssim_list, psnr_list, mse_list, metrics_list = [], [], [], [], []
        for idx in range(input_.shape[0]):
            _pcc, _ssim, _psnr, _mse, metrics = self._metric(target_[idx].cpu().numpy(), output_[idx].cpu().numpy())
            pcc_list.append(_pcc)
            ssim_list.append(_ssim)
            psnr_list.append(_psnr)
            mse_list.append(_mse)
            metrics_list.append(metrics)
        results = {"val_loss": loss, "pcc": pcc_list, "ssim": ssim_list, "psnr": psnr_list, "mse": mse_list,  "metrics": metrics_list}
        self.validation_step_outputs.append(results)
        return results
    
    def on_validation_epoch_end(self) -> None:
        loss_list, loss_all_list, loss_tissue_list, pcc_list, ssim_list, psnr_list, mse_list, metrics_list = [], [], [], [], [], [], [], []
        for output in self.validation_step_outputs:
            loss_list.append(output["val_loss"].item())
            # loss_all_list.append(output["val_mse"].item())
            # loss_tissue_list.append(output["val_tissue_mse"].item())
            pcc_list.extend(output["pcc"])
            ssim_list.extend(output["ssim"])
            psnr_list.extend(output["psnr"])
            mse_list.extend(output["mse"])
            metrics_list.extend(output["metrics"])
        self.validation_step_outputs.clear()
        self.log("valid/loss", np.mean(loss_list), on_epoch=True, sync_dist=True)
        # self.log("valid/loss_mse", np.mean(loss_all_list), on_epoch=True, sync_dist=True)
        # self.log("valid/loss_tissue_mse", np.mean(loss_tissue_list), on_epoch=True, sync_dist=True)
        self.log("valid/pcc", np.mean(pcc_list), on_epoch=True, sync_dist=True)
        self.log("valid/ssim", np.mean(ssim_list), on_epoch=True, sync_dist=True)
        self.log("valid/psnr", np.mean(psnr_list), on_epoch=True, sync_dist=True)
        self.log("valid/mse", np.mean(mse_list), on_epoch=True, sync_dist=True)
        self.log("valid/metrics", np.mean(metrics_list), on_epoch=True, sync_dist=True)


    def test_step(self, batch, batch_idx: int) -> dict:
        input_, target_, bone_mask_, tissue_mask_, metal_size_, imgName = batch
        output_ = self.step(input_)
        loss = self.criterion(output_, target_, tissue_mask_)
        pcc_list, ssim_list, psnr_list, mse_list, metal_list = [], [], [], [], []
        for idx in range(input_.shape[0]):
            _pcc, _ssim, _psnr, _mse, _ = self._metric(target_[idx].cpu().numpy(), output_[idx].cpu().numpy())
            pcc_list.append(_pcc)
            ssim_list.append(_ssim)
            psnr_list.append(_psnr)
            mse_list.append(_mse)
            metal_list.append(metal_size_[idx].cpu().numpy())
            if self.save_output_only:
                save_as_dicom_test(output_=output_[idx].cpu().numpy(),
                              test_save_path=self.test_save_path,
                              imgName=imgName[idx],
                              save_path=self.save_path
                              )
            else:
                save_as_dicom_test(input_=input_[idx].cpu().numpy(), 
                              target_=target_[idx].cpu().numpy(), 
                              output_=output_[idx].cpu().numpy(), 
                              test_save_path=self.test_save_path, 
                              imgName=imgName[idx],
                              save_path=self.save_path)
        results = {"loss":loss, "pcc": pcc_list, "ssim": ssim_list, "psnr": psnr_list, "mse": mse_list, "metal_size": metal_list, "imgName": imgName}
        self.testing_step_outputs.append(results)
        return results
    
    def on_test_epoch_end(self) -> None:
        loss_list, pcc_list, ssim_list, psnr_list, mse_list, metal_list, name_list = [], [], [], [], [], [], []
        for output in self.testing_step_outputs:
            loss_list.append(output["loss"].item())
            pcc_list.extend(output["pcc"])
            ssim_list.extend(output["ssim"])
            psnr_list.extend(output["psnr"])
            mse_list.extend(output["mse"])
            metal_list.extend(output["metal_size"])
            name_list.extend(output["imgName"])
        print("------------------")
        print("Evaluation Result")
        print(f"pcc: {np.mean(pcc_list)}")
        print(f"ssim: {np.mean(ssim_list)}")
        print(f"psnr: {np.mean(psnr_list)}")
        print(f"mse: {np.mean(mse_list)}")
        df = pd.DataFrame({"fname": name_list, "pcc":pcc_list, "ssim":ssim_list, "psnr":psnr_list, "mse":mse_list, "metal_size": metal_list})
        df.to_csv(os.path.join(self.test_save_path, "test_dcm", self.save_path.split('/')[-3]+".csv"))
        
    def predict_step(self, batch, batch_idx) -> None:
        input_, imgName = batch
        output_ = self.step(input_)
        for idx in range(input_.shape[0]):
            save_as_dicom_infer(output_=output_[idx], 
                                test_save_path=self.test_save_path, 
                                imgName=imgName[idx],
                                save_path=self.save_path
                                )
        
