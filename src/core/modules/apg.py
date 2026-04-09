import torch


class APG:
    def __init__(self, guide_scale: float, momentum: float, eta: float = 1.0, norm_threshold: float = 0.0):
        self.momentum = momentum
        self.running_average = 0
        self.guide_scale = guide_scale
        self.eta = eta
        self.norm_threshold = norm_threshold
        
    def update_momentum(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average
        
    def project(self, v0: torch.Tensor, v1: torch.Tensor):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)
    
    def adaptive_projected_guidance(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor):
        diff = pred_cond - pred_uncond
        
        self.update_momentum(diff)
        diff = self.running_average
            
        if self.norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True)
            scale_factor = torch.minimum(ones, self.norm_threshold / diff_norm)
            diff = diff * scale_factor
            
        diff_parallel, diff_orthogonal = self.project(diff, pred_cond)
        normalized_update = diff_orthogonal + self.eta * diff_parallel
        pred_guided = pred_cond + (self.guide_scale - 1) * normalized_update
        return pred_guided
    
    def get_apg_noise_pred(
        self,
        xt: torch.Tensor, 
        noise_pred_cond: torch.Tensor, 
        noise_pred_uncond: torch.Tensor, 
        t: torch.Tensor
    ):
        t = t.view(-1, 1, 1, 1, 1) / 1000
        pred_cond = xt - noise_pred_cond * t
        pred_uncond = xt - noise_pred_uncond * t
        
        pred_guided = self.adaptive_projected_guidance(pred_cond, pred_uncond)
        noise_pred_guided = (xt - pred_guided) / t
        return noise_pred_guided
