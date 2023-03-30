
import numpy as np
import torch
from torch import nn
from einops import (rearrange, reduce, repeat)

from .utils.grid_sample import grid_sample_2d
from .utils.cnn3d import VolumeRegularization

class FeatureVolume(nn.Module):
    """
    Create the coarse feature volume in a MVS-like way
    """
    def __init__(self, volume_reso):
        """
        Set up the volume grid given resolution
        """
        super().__init__()
        
        self.volume_reso = volume_reso
        self.volume_regularization = VolumeRegularization()

        # the volume is a cube, so we only need to define the x, y, z
        x_line = (np.linspace(0,self.volume_reso-1,self.volume_reso))*2/(self.volume_reso-1) - 1 # [-1, 1]
        y_line = (np.linspace(0,self.volume_reso-1,self.volume_reso))*2/(self.volume_reso-1) - 1
        z_line = (np.linspace(0,self.volume_reso-1,self.volume_reso))*2/(self.volume_reso-1) - 1

        # create the volume grid
        self.x, self.y, self.z = np.meshgrid(x_line, y_line, z_line, indexing='ij')
        self.xyz = np.stack([self.x, self.y, self.z])

        self.linear = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8)
        )


    def forward(self, feats, batch):
        """
        feats: [B NV C H W], NV: number of views
        batch: to get the poses for homography
        """
        source_poses = batch['source_poses']
        B, NV, _, _ = source_poses.shape

        # ---- step 1: projection -----------------------------------------------
        volume_xyz = torch.tensor(self.xyz).type_as(source_poses)
        volume_xyz = volume_xyz.reshape([3,-1])
        volume_xyz_homo = torch.cat([volume_xyz, torch.ones_like(volume_xyz[0:1])], axis=0)  #[4,XYZ]

        volume_xyz_homo_NV = repeat(volume_xyz_homo, "Num4 XYZ -> B NV Num4 XYZ", B=B, NV=NV)

        # volume project into views
        volume_xyz_pixel_homo = source_poses @ volume_xyz_homo_NV # B NV 4 4 @ B NV 4 XYZ
        volume_xyz_pixel_homo = volume_xyz_pixel_homo[:,:,:3]
        mask_valid_depth = volume_xyz_pixel_homo[:,:,2]>0  #B NV XYZ
        mask_valid_depth = mask_valid_depth.float()
        mask_valid_depth = rearrange(mask_valid_depth, "B NV XYZ -> (B NV) XYZ")

        volume_xyz_pixel = volume_xyz_pixel_homo / volume_xyz_pixel_homo[:,:,2:3]
        volume_xyz_pixel = volume_xyz_pixel[:,:,:2]
        volume_xyz_pixel = rearrange(volume_xyz_pixel, "B NV Dim2 XYZ -> (B NV) XYZ Dim2")
        volume_xyz_pixel = volume_xyz_pixel.unsqueeze(2)

        # projection: project all x * y * z points to NV images and sample features

        # grid sample 2D 
        volume_feature, mask = grid_sample_2d(rearrange(feats, "B NV C H W -> (B NV) C H W"), volume_xyz_pixel) # (B NV) C XYZ 1, (B NV XYZ 1)
        
        volume_feature = volume_feature.squeeze(-1)
        mask = mask.squeeze(-1)  # (B NV XYZ)
        mask = mask * mask_valid_depth

        volume_feature = rearrange(volume_feature, "(B NV) C (NumX NumY NumZ) -> B NV NumX NumY NumZ C", B=B, NV=NV, NumX=self.volume_reso, NumY=self.volume_reso, NumZ=self.volume_reso)
        mask = rearrange(mask, "(B NV) (NumX NumY NumZ) -> B NV NumX NumY NumZ", B=B, NV=NV, NumX=self.volume_reso, NumY=self.volume_reso, NumZ=self.volume_reso)

        weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        weight = weight.unsqueeze(-1)  # B NV X Y Z 1

        # ---- step 2: compress ------------------------------------------------
        volume_feature_compressed = self.linear(volume_feature)

        # ---- step 3: mean, var ------------------------------------------------
        mean = torch.sum(volume_feature_compressed * weight, dim=1, keepdim=True)  #B 1 X Y Z C
        var = torch.sum(weight * (volume_feature_compressed - mean)**2, dim=1, keepdim=True)  # B 1 X Y Z C
        mean = mean.squeeze(1)
        var = var.squeeze(1)

        volume_mean_var = torch.cat([mean, var], axis=-1)  #[B X Y Z C]
        volume_mean_var = volume_mean_var.permute(0,4,3,2,1)  #[B,C,Z,Y,X]

        # ---- step 4: 3D regularization ----------------------------------------
        volume_mean_var_reg = self.volume_regularization(volume_mean_var)

        return volume_mean_var_reg