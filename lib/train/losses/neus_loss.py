import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import net_utils
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mask_weight = cfg.train.mask_weight
        self.igr_weight = cfg.train.igr_weight
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def forward(self, batch):
        render_out = self.net(batch)

        color_fine = render_out['color_fine']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        true_rgb = batch['color']
        mask = batch['mask']
        if self.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        mask_sum = mask.sum() + 1e-5
        # Loss
        color_error = (color_fine - true_rgb.squeeze(0)) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        eikonal_loss = gradient_error

        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask.squeeze(0))

        loss = color_fine_loss + \
               eikonal_loss * self.igr_weight + \
               mask_loss * self.mask_weight

        scalar_stats = {}
        # loss = 0
        color_loss = self.color_crit(render_out['color_fine'], batch['color'])
        scalar_stats.update({'color_mse': color_loss})
        # loss += color_loss

        psnr = -10. * torch.log(color_loss.detach()) / torch.log(torch.Tensor([10.]).to(color_loss.device))
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return render_out, loss, scalar_stats, image_stats
