import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import (rearrange, reduce, repeat)

from .utils.grid_sample import grid_sample_2d, grid_sample_3d
from .attention.transformer import LocalFeatureTransformer

import math
PI = math.pi

class PositionEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        self.augmented = rearrange((PI * 2 ** torch.arange(-1, self.L - 1)), "L -> L 1 1 1")

    def forward(self, x):
        sin_term = torch.sin(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim")) # BUG? 
        cos_term = torch.cos(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim") )
        sin_cos_term = torch.stack([sin_term, cos_term])

        sin_cos_term = rearrange(sin_cos_term, "Num2 L RN SN Dim -> (RN SN) (L Num2 Dim)")

        return sin_cos_term


class RayTransformer(nn.Module):
    """
    Ray transformer
    """
    def __init__(self, args, img_feat_dim=32, fea_volume_dim=16):
        super().__init__()

        self.args = args
        self.offset =  [[0, 0, 0]]

        self.volume_reso = args.volume_reso
        self.only_volume = False
        if self.only_volume:
            assert self.volume_reso > 0, "if only use volume feature, must have volume"

        self.img_feat_dim = img_feat_dim
        self.fea_volume_dim = fea_volume_dim if self.volume_reso > 0 else 0
        
        self.PE_d_hid = 8

        # transformers
        self.density_view_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.fea_volume_dim, 
                                    nhead=8, layer_names=['self'], attention='linear')

        self.density_ray_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.PE_d_hid + self.fea_volume_dim, 
                                    nhead=8, layer_names=['self'], attention='linear')

        if self.only_volume:
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.fea_volume_dim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))
        else:
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.img_feat_dim + self.PE_d_hid + self.fea_volume_dim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))

        self.relu = nn.ReLU(inplace=True)

        # learnable view token
        self.viewToken = ViewTokenNetwork(dim=self.img_feat_dim + self.fea_volume_dim)
        self.softmax = nn.Softmax(dim=-2)

        # to calculate radiance weight
        self.linear_radianceweight_1_softmax = nn.Sequential(
            nn.Linear(self.img_feat_dim+3+self.fea_volume_dim, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )


    def order_posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)

        return sinusoid_table


    def forward(self, point3D, batch, source_imgs_feat, fea_volume=None):

        B, NV, _, H, W = batch['source_imgs'].shape
        _, RN, SN, _ = point3D.shape
        FDim = source_imgs_feat.size(2) # feature dim
        CN = len(self.offset)

        # calculate relative direction
        vector_1 = (point3D - repeat(batch['ref_pose_inv'][:,:3,-1], "B DimX -> B 1 1 DimX"))
        vector_1 = repeat(vector_1, "B RN SN DimX -> B 1 RN SN DimX")
        vector_2 = (point3D.unsqueeze(1) - repeat(batch['source_poses_inv'][:,:,:3,-1], "B L DimX -> B L 1 1 DimX")) # B L RN SN DimX
        vector_1 = vector_1/torch.linalg.norm(vector_1, dim=-1, keepdim=True) # normalize to get direction
        vector_2 = vector_2/torch.linalg.norm(vector_2, dim=-1, keepdim=True)
        dir_relative = vector_1 - vector_2 
        dir_relative = dir_relative.float()

        if self.args.volume_reso > 0: 
            assert fea_volume != None
            fea_volume_feat = grid_sample_3d(fea_volume, point3D.unsqueeze(1).float())
            fea_volume_feat = rearrange(fea_volume_feat, "B C RN SN -> (B RN SN) C")
        # -------- project points to feature map
        # B NV RN SN CN DimXYZ
        point3D = repeat(point3D, "B RN SN DimX -> B NV RN SN DimX", NV=NV).float()
        point3D = torch.cat([point3D, torch.ones_like(point3D[:,:,:,:,:1])], axis=4)
        
        # B NV 4 4 -> (B NV) 4 4
        points_in_pixel = torch.bmm(rearrange(batch['source_poses'], "B NV M_1 M_2 -> (B NV) M_1 M_2", M_1=4, M_2=4), 
                                rearrange(point3D, "B NV RN SN DimX -> (B NV) DimX (RN SN)"))
        
        points_in_pixel = rearrange(points_in_pixel, "(B NV) DimX (RN SN) -> B NV DimX RN SN", B=B, RN=RN)
        points_in_pixel = points_in_pixel[:,:,:3]
        # in 2D pixel coordinate
        mask_valid_depth = points_in_pixel[:,:,2]>0  #B NV RN SN
        mask_valid_depth = mask_valid_depth.float()
        points_in_pixel = points_in_pixel[:,:,:2] / points_in_pixel[:,:,2:3]

        img_feat_sampled, mask = grid_sample_2d(rearrange(source_imgs_feat, "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"))
        img_rgb_sampled, _ = grid_sample_2d(rearrange(batch['source_imgs'], "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"))

        mask = rearrange(mask, "(B NV) RN SN -> B NV RN SN", B=B)
        mask = mask * mask_valid_depth
        img_feat_sampled = rearrange(img_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)
        img_rgb_sampled = rearrange(img_rgb_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)

        # --------- run transformer to aggregate information
        # -- 1. view transformer
        x = rearrange(img_feat_sampled, "B NV C RN SN -> (B RN SN) NV C")
        
        if self.args.volume_reso > 0: 
            x_fea_volume_feat = repeat(fea_volume_feat, "B_RN_SN C -> B_RN_SN NV C", NV=NV)
            x = torch.cat([x, x_fea_volume_feat], axis=-1)

        # add additional view aggregation token
        view_token = self.viewToken(x)
        view_token = rearrange(view_token, "B_RN_SN C -> B_RN_SN 1 C")
        x = torch.cat([view_token, x], axis=1)
        x = self.density_view_transformer(x)

        x1 = rearrange(x, "B_RN_SN NV C -> NV B_RN_SN C")
        x = x1[0] #reference
        view_feature = x1[1:]

        if self.only_volume:
            x = rearrange(x_fea_volume_feat, "(B RN SN) NV C -> NV (B RN) SN C", B=B, RN=RN, SN=SN)[0]
        else:
            # -- 2. ray transformer
            # add positional encoding
            x = rearrange(x, "(B RN SN) C -> (B RN) SN C", RN=RN, B=B, SN=SN)
            x = torch.cat([x, repeat(self.order_posenc(d_hid=self.PE_d_hid, n_samples=SN).type_as(x), 
                                        "SN C -> B_RN SN C", B_RN = B*RN)], axis=2)
            x = self.density_ray_transformer(x)        

        srdf = self.DensityMLP(x)

        # calculate weight using view transformers result
        view_feature = rearrange(view_feature, "NV (B RN SN) C -> B RN SN NV C", B=B, RN=RN, SN=SN)
        dir_relative = rearrange(dir_relative, "B NV RN SN Dim3 -> B RN SN NV Dim3")

        x_weight = torch.cat([view_feature, dir_relative], axis=-1)
        x_weight = self.linear_radianceweight_1_softmax(x_weight)
        mask = rearrange(mask, "B NV RN SN -> B RN SN NV 1")
        x_weight[mask==0] = -1e9
        weight = self.softmax(x_weight)
        
        radiance = (img_rgb_sampled * rearrange(weight, "B RN SN L 1 -> B L 1 RN SN", B=B, RN=RN)).sum(axis=1)
        radiance = rearrange(radiance, "B DimRGB RN SN -> (B RN SN) DimRGB")

        return radiance, srdf, points_in_pixel  


class ViewTokenNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_parameter('view_token', nn.Parameter(torch.randn([1,dim])))

    def forward(self, x):
        return torch.ones([len(x), 1]).type_as(x) * self.view_token
