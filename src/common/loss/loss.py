import torch
from . import ssim_loss

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        if torch.sum(mask)==0:
            diff2 = ((input) - (target)) ** 2.0
            result = torch.mean(diff2)
        else:
            diff2 = ((input) - (target)) ** 2.0 * mask
            result = torch.sum(diff2) / torch.sum(mask)
        return result
    
class MSE_SSIM_Loss(torch.nn.Module):
    def __init__(self, ssim_rate: float=0.1, device: str=None) ->None:
        super(MSE_SSIM_Loss, self).__init__()
        self.recon_loss = torch.nn.MSELoss()
        self.ssim_loss = ssim_loss.SSIM(torch_device=device)
        self.ssim_rate = ssim_rate

    def forward(self, output: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        loss1 = self.recon_loss(output, target)
        loss2 = self.ssim_loss(output, target)
        loss = loss1 + self.ssim_rate * loss2 
        return loss

class Masked_MSE_SSIM_Loss(torch.nn.Module):
    def __init__(self, ssim_rate: float=0.1, tissue_rate: float=10., device: str=None) ->None:
        super(Masked_MSE_SSIM_Loss, self).__init__()
        self.recon_loss = torch.nn.MSELoss()
        self.ssim_loss = ssim_loss.SSIM(torch_device=device)
        self.ssim_rate = ssim_rate
        self.tissue_rate = tissue_rate

    def forward(self, output: torch.Tensor, target: torch.Tensor, tissue_mask: torch.Tensor) ->torch.Tensor:
        loss1_bone1 = self.recon_loss(output, target)
        loss1_bone2 = self.recon_loss(output* tissue_mask, target* tissue_mask)

        loss1 = loss1_bone1 + self.tissue_rate * loss1_bone2    
        loss2 = self.ssim_loss(output, target) #* self.ssim_rate     # MSE + ssim loss
        loss = loss1 + self.ssim_rate * loss2
        return loss