import torch

class VolumeRenderer():
    def __init__(self, args=None):
        self.args = args

    def render(self, z_val, radiance, geo_value, cos_anneal_ratio=1.0, deviation_network=None): 
        """
        Volume rendering pixels given srdf and radiance of samples
        Adapted from: https://github.com/xxlong0/SparseNeuS 

        z_val: z value of each sample, [RN, SN]
        radiance: radiance of each sample, [RN, SN, 3]
        geo_value: geo value of each sample, [RN, SN] (srdf)
        cos_anneal_ratio: cosine annealing ratio
        deviation_network: network to predict deviation
        """

        interval = z_val[:,1:]-z_val[:,:-1]
        interval = torch.cat([interval[:,0:1], interval, interval[:,-1:]], axis=1)
        interval = (interval[:,:-1] + interval[:,1:]) / 2

        batch_size, n_samples = z_val.shape 
        srdf = geo_value
        inv_s0 = deviation_network(torch.zeros([1, 3]).type_as(z_val))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s0.expand(batch_size, n_samples)

        true_cos = -1.0
        iter_cos = -(-true_cos * 0.5 + 0.5 * (1.0 - cos_anneal_ratio) -true_cos * cos_anneal_ratio)  

        estimated_next_srdf = srdf + iter_cos * interval * 0.5
        estimated_prev_srdf = srdf - iter_cos * interval * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_srdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_srdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        weight = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).type_as(z_val), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        rgb = (radiance * weight[:,:,None]).sum(axis=1)
        depth = (weight * z_val).sum(axis=1)
        opacity = weight.sum(axis=1)
        
        return rgb, depth, opacity, weight, 1.0 / inv_s0
