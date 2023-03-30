import os, piq
from re import I
from stat import UF_OPAQUE

import numpy as np
import torch
from torch import optim
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl

from einops import (rearrange, reduce, repeat)

from .utils.sampler import FixedSampler, ImportanceSampler
from .utils.feature_extractor import FPN_FeatureExtractor
from .utils.single_variance_network import SingleVarianceNetwork
from .utils.renderer import VolumeRenderer

from .ray_transformer import RayTransformer
from .feature_volume import FeatureVolume


class VolRecon(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.train_ray_num = args.train_ray_num

        if self.args.extract_geometry: # testing
            self.point_num = args.test_sample_coarse
            self.point_num_2 = args.test_sample_fine
        else:
            self.point_num = args.coarse_sample
            self.point_num_2 = args.fine_sample

        self.feat_extractor = FPN_FeatureExtractor(out_ch=32)
        self.fixed_sampler = FixedSampler(point_num = self.point_num)
        self.importance_sampler = ImportanceSampler(point_num = self.point_num_2)
        self.deviation_network = SingleVarianceNetwork(0.3) # add variance network

        self.renderer = VolumeRenderer(args)
        self.ray_transformer = RayTransformer(args = args) 

        if self.args.volume_reso>0:
            self.feature_volume = FeatureVolume(self.args.volume_reso)

        self.pos_encoding = self.order_posenc(d_hid=16, n_samples=self.point_num)
        self.pos_encoding_2 = self.order_posenc(d_hid=16, n_samples=self.point_num + self.point_num_2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer
    

    def order_posenc(self, d_hid, n_samples):
        """
        positional encoding of the sample ordering on a ray
        """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)
        
        return sinusoid_table


    def build_feature_volume(self, batch, source_imgs_feat):
        return self.feature_volume(source_imgs_feat, batch)


    def sample2rgb(self, batch, points_x, z_val, ray_d, ray_idx, source_imgs_feat, feature_volume):
        B, L, _, imgH, imgW = batch['source_imgs'].shape
        RN = ray_idx.shape[1]
        _, _, SN, _ = points_x.shape

        radiance, srdf, points_in_pixel = self.ray_transformer(points_x, batch, source_imgs_feat, feature_volume)

        ray_d = repeat(ray_d, "RN Dim3 -> RN SN Dim3", SN=SN)

        rgb, depth, opacity, weight, variance = self.renderer.render(rearrange(z_val, "B RN SN -> (B RN) SN"),  
                                        rearrange(radiance, "(B RN SN) C -> (B RN) SN C", B=B, RN=RN),
                                        srdf.squeeze(dim=2),
                                        deviation_network=self.deviation_network)

        rgb = rearrange(rgb, "(B RN) C -> B RN C", B=B).float()
        depth = rearrange(depth, "(B RN) -> B RN", B=B)
        opacity = rearrange(opacity, "(B RN) -> B RN", B=B)
        weight = rearrange(weight, "(B RN) SN -> B RN SN", B=B)

        return rgb, depth, srdf, opacity, weight, points_in_pixel, variance


    def infer(self, batch, ray_idx, source_imgs_feat, feature_volume=None, extract_geometry=False):

        B, L, _, imgH, imgW = batch['source_imgs'].shape
        RN = ray_idx.shape[1]
        
        if not extract_geometry:
            # gt rgb for rays
            ref_img = rearrange(batch['ref_img'], "B DimRGB H W -> B DimRGB (H W)")
            rgb_gt = torch.gather(ref_img, 2, repeat(ray_idx, "B RN -> B DimRGB RN", DimRGB=3))
            rgb_gt = rearrange(rgb_gt, "B C RN -> B RN C")

            # gt depth for rays
            ref_depth = rearrange(batch['depths_h'][:,0], "B H W -> B (H W)") # use only depth of reference view 
            depth_gt = torch.gather(ref_depth, 1, ray_idx)
        
        ray_d = torch.gather(batch['ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
        ray_d = rearrange(ray_d, "B DimX RN -> (B RN) DimX")
        ray_o = repeat(batch['ray_o'], "B DimX -> B DimX RN", RN = RN) 
        ray_o = rearrange(ray_o, "B DimX RN -> (B RN) DimX")

        # ---------------------- coarse sampling along the ray ----------------------
        if 'near_fars' in batch.keys():
            near_z = batch['near_fars'][:,0,0]
            near_z = repeat(near_z, "B -> B RN", RN=RN)
            near_z = rearrange(near_z, "B RN -> (B RN)")
            far_z = batch['near_fars'][:,0,1]
            far_z = repeat(far_z, "B -> B RN", RN=RN)
            far_z = rearrange(far_z, "B RN -> (B RN)")

            if extract_geometry:
                camera_ray_d = torch.gather(batch['cam_ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
                camera_ray_d = rearrange(camera_ray_d, "B DimX RN -> (B RN) DimX")
                near_z = near_z / camera_ray_d[:,2]
                far_z = far_z / camera_ray_d[:,2]
            points_x, z_val, points_d = self.fixed_sampler.sample_ray(ray_o, ray_d, near_z=near_z, far_z=far_z)

        else:
            points_x, z_val, points_d = self.fixed_sampler.sample_ray(ray_o, ray_d)

        # SN is sample point number along the ray
        points_x = rearrange(points_x, "(B RN) SN DimX -> B RN SN DimX", B = B) 
        points_d = rearrange(points_d, "(B RN) SN DimX -> B RN SN DimX", B = B)

        z_val = rearrange(z_val, "(B RN) SN -> B RN SN", B = B)

        rgb, depth, srdf, opacity, weight, points_in_pixel, _ = self.sample2rgb(batch, points_x, z_val, ray_d, ray_idx, 
                    source_imgs_feat, feature_volume=feature_volume)

        if extract_geometry and self.args.test_coarse_only:
            srdf = rearrange(srdf, "(B RN) SN Dim1 ->B RN SN Dim1", B=B)
            srdf = srdf.squeeze(-1)
            return srdf, points_x, depth, rgb

        # ---------------------- fine sampling along the ray ----------------------
        points_x_2, z_val_2, points_d_2 = self.importance_sampler.sample_ray(ray_o, ray_d, 
                                                        rearrange(weight, "B RN SN -> (B RN) SN", B=B).detach(), 
                                                        rearrange(z_val, "B RN SN -> (B RN) SN").detach())
        
        # SN is sample point number along the ray
        points_x_2 = rearrange(points_x_2, "(B RN) SN DimX -> B RN SN DimX", B = B) 
        points_d_2 = rearrange(points_d_2, "(B RN) SN DimX -> B RN SN DimX", B = B)
        z_val_2 = rearrange(z_val_2, "(B RN) SN -> B RN SN", B = B)

        points_x_all = torch.cat([points_x, points_x_2], axis=2)
        z_val_all = torch.cat([z_val, z_val_2], axis=2)
        sample_sort_idx = torch.sort(z_val_all,axis=2)[1]
        z_val_all = torch.gather(z_val_all, 2, sample_sort_idx)
        points_x_all = torch.gather(points_x_all, 2, repeat(sample_sort_idx, "B RN SN -> B RN SN 3"))

        rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2, variance = self.sample2rgb(batch, 
        points_x_all, z_val_all, ray_d, ray_idx, source_imgs_feat, feature_volume=feature_volume)

        if extract_geometry:
            srdf_2 = rearrange(srdf_2, "(B RN) SN Dim1 ->B RN SN Dim1", B=B)
            srdf_2 = srdf_2.squeeze(-1)
            return srdf_2, points_x_all, depth_2, rgb_2

        return rgb_gt, rgb, depth, depth_gt, srdf, opacity, weight, points_in_pixel,\
            rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2,\
            z_val, z_val_all, variance


    def training_step(self, batch, batch_idx):
        B, L, _, imgH, imgW = batch['source_imgs'].shape
        
        # ---------------------- step 0: infer image features ----------------------
        source_imgs = rearrange(batch['source_imgs'], "B L C H W -> (B L) C H W")
        source_imgs_feat = self.feat_extractor(source_imgs)

        source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L)

        if self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, source_imgs_feat)
        else:
            feature_volume = None

        # ---------------------- step 1: sample rays --------------------------------
        ray_idx = torch.argsort(torch.rand(B, imgH * imgW).type_as(batch['ray_o']), dim=-1)[:,:self.train_ray_num]

        rgb_gt, rgb, depth, depth_gt, srdf, opacity, weight, points_in_pixel, \
            rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2, \
            z_val, z_val_all, variance = self.infer(batch=batch, 
                                                ray_idx=ray_idx, 
                                                source_imgs_feat=source_imgs_feat,
                                                feature_volume=feature_volume)

        B, _ = depth_gt.size()

        # color loss
        loss_rgb = torch.nn.functional.mse_loss(rgb, rgb_gt)
        loss_rgb2 = torch.nn.functional.mse_loss(rgb_2, rgb_gt)

        # Depth loss
        mask_depth = (depth_gt!=0) & (depth_gt>=batch['near_fars'][:,0,0:1]) & (depth_gt<=batch['near_fars'][:,0,1:2])

        if torch.sum(mask_depth)>0:
            # masked out where gt depth is invalid
            loss_depth_ray =  torch.nn.functional.l1_loss(depth[mask_depth], depth_gt[mask_depth]) 
            loss_depth_ray2 = torch.nn.functional.l1_loss(depth_2[mask_depth], depth_gt[mask_depth])
        else:
            loss_depth_ray = loss_depth_ray2 = 0.0

        loss = self.args.weight_rgb * (loss_rgb + loss_rgb2) + \
                self.args.weight_depth * (loss_depth_ray + loss_depth_ray2)

        self.log("train/depth_ray_coarse", loss_depth_ray)
        self.log("train/depth_ray_fine", loss_depth_ray2)
        self.log("train/rgb_coarse", loss_rgb)
        self.log("train/rgb_fine", loss_rgb2)
        self.log("train/loss_all", loss)
        self.log("train/variance", variance)

        return loss


    def validation_epoch_end(self, batch_parts):
        # average epoches

        psnr_coarse = [i['psnr/coarse'] for i in batch_parts]
        psnr_fine = [i['psnr/fine'] for i in batch_parts]
        loss_rgb_coarse = [i['val/loss_rgb_coarse'] for i in batch_parts]
        loss_rgb_fine = [i['val/loss_rgb_fine'] for i in batch_parts]
        loss_depth_coarse = [i['val/loss_depth_coarse'] for i in batch_parts]
        loss_depth_fine = [i['val/loss_depth_fine'] for i in batch_parts]

        psnr_coarse = sum(psnr_coarse) / len(psnr_coarse)
        psnr_fine = sum(psnr_fine) / len(psnr_fine)
        loss_rgb_coarse = sum(loss_rgb_coarse) / len(loss_rgb_coarse)
        loss_rgb_fine = sum(loss_rgb_fine) / len(loss_rgb_fine)
        loss_depth_coarse = sum(loss_depth_coarse) / len(loss_depth_coarse)
        loss_depth_fine = sum(loss_depth_fine) / len(loss_depth_fine)
        
        # logging
        self.log("psnr/coarse", psnr_coarse, sync_dist=True)
        self.log("psnr/fine", psnr_fine, sync_dist=True)
        self.log("val/rgb_coarse", loss_rgb_coarse, sync_dist=True)
        self.log("val/rgb_fine", loss_rgb_fine, sync_dist=True)
        self.log("val/loss_depth_coarse", loss_depth_coarse, sync_dist=True)
        self.log("val/loss_depth_fine", loss_depth_fine, sync_dist=True)

        loss = loss_rgb_coarse + loss_rgb_fine

        return loss


    def validation_step(self, batch, batch_idx):
        if self.args.extract_geometry:
            self.extract_geometry(batch, batch_idx)

            # return dummy data
            return {"val/loss_rgb_coarse":0,
                    "val/loss_rgb_fine":0,
                    "val/loss_depth_coarse":0,
                    "val/loss_depth_fine":0,
                    "psnr/coarse":0,
                    "psnr/fine":0}

        B, L, _, imgH, imgW = batch['source_imgs'].shape

        # ---------------------- step 0: infer image features --------------------------------
        source_imgs = rearrange(batch['source_imgs'], "B L C H W -> (B L) C H W")
        source_imgs_feat = self.feat_extractor(source_imgs)
        source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L)
        
        # ---------------------- step 1: sample rays -----------------------------------------
        ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long() 

        rgb_list, rgb_gt_list, depth_list, rgb_list_2, depth_list_2 = [], [], [], [], []

        if self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, source_imgs_feat)
        else:
            feature_volume = None

        for ray_idx in tqdm(torch.split(ray_idx_all, self.train_ray_num, dim=1)):
            rgb_gt, rgb, depth, _, _, _, _, _, \
                rgb_2, depth_2, _, _, _, _, _, _, variance = \
                        self.infer(batch=batch, ray_idx=ray_idx, source_imgs_feat=source_imgs_feat, feature_volume=feature_volume)

            rgb_list.append(rgb)
            rgb_gt_list.append(rgb_gt)
            depth_list.append(depth)
            rgb_list_2.append(rgb_2)
            depth_list_2.append(depth_2)

        rgb_list = torch.cat(rgb_list, dim=1)
        rgb_gt_list = torch.cat(rgb_gt_list, axis=1)
        depth_list = torch.cat(depth_list, axis=1)
        rgb_list_2 = torch.cat(rgb_list_2, dim=1)
        depth_list_2 = torch.cat(depth_list_2, axis=1)

        # move to cpu
        to_CPU = lambda x: x.cpu().numpy()
        variance = to_CPU(variance)

        rgb_imgs = rearrange(rgb_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        rgb_gt_imgs = rearrange(rgb_gt_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        depths = rearrange(depth_list, "B (H W) -> B H W", H=imgH)
        rgb_imgs_2 = rearrange(rgb_list_2, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        depths_2 = rearrange(depth_list_2, "B (H W) -> B H W", H=imgH)
        
        # metrics
        loss_rgb = torch.nn.functional.mse_loss(rgb_list, rgb_gt_list)
        loss_rgb_2 = torch.nn.functional.mse_loss(rgb_list_2, rgb_gt_list)

        psnr_coarse = piq.psnr(torch.clamp(rgb_imgs, max=1, min=0), torch.clamp(rgb_gt_imgs, max=1, min=0)).item()
        psnr_fine = piq.psnr(torch.clamp(rgb_imgs_2, max=1, min=0), torch.clamp(rgb_gt_imgs, max=1, min=0)).item()

        # return depth loss and log it
        depth_gt = batch['depths_h'][:,0]
        
        # Depth loss
        B,H,W = depth_gt.size()
        mask_depth = (depth_gt!=0) & (depth_gt>=batch['near_fars'][:,0:1,0:1]) & (depth_gt<=batch['near_fars'][:,0:1,1:2])

        if torch.sum(mask_depth)>0:
            # masked out where gt depth is invalid
            loss_depth_ray =  torch.nn.functional.l1_loss(depths[mask_depth], depth_gt[mask_depth]) 
            loss_depth_ray2 = torch.nn.functional.l1_loss(depths_2[mask_depth], depth_gt[mask_depth])
        else:
            loss_depth_ray = loss_depth_ray2 = 0.0

        return {"val/loss_rgb_coarse":loss_rgb.item(), 
                "val/loss_rgb_fine":loss_rgb_2.item(), 
                "val/loss_depth_coarse":loss_depth_ray.item(), 
                "val/loss_depth_fine":loss_depth_ray2.item(), 
                "psnr/coarse":psnr_coarse, 
                "psnr/fine":psnr_fine,
                "val/variance": variance}


    def extract_geometry(self, batch, batch_idx):
        
        B, L, _, imgH, imgW = batch['source_imgs'].shape
        scan_name = batch['meta'][0].split("-")[1]
        ref_view = batch['meta'][0].split("-")[-1]
        os.makedirs(os.path.join(self.args.out_dir, scan_name, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.args.out_dir, "depth", scan_name), exist_ok=True)
        os.makedirs(os.path.join(self.args.out_dir, "rgb", scan_name), exist_ok=True)

        print("strat extracting geometry")
        source_imgs_feat = []
        for l in range(L):
            source_img = batch['source_imgs'][:,l]
            feat = self.feat_extractor(source_img)
            source_imgs_feat.append(feat)
        source_imgs_feat = torch.stack(source_imgs_feat, dim=1)

        ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long()
        depth_list, rgb_list  = [], []

        if self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, source_imgs_feat)
        else:
            feature_volume = None

        for ray_idx in tqdm(torch.split(ray_idx_all, self.args.test_ray_num, dim=1)):
            srdf, points_x, depth, rgb = self.infer(batch=batch, ray_idx=ray_idx, source_imgs_feat=source_imgs_feat, 
                                feature_volume=feature_volume, extract_geometry=True)

            ray_d = torch.gather(batch['cam_ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
            ray_d = rearrange(ray_d, "B DimX RN -> B RN DimX")

            depth = (depth.unsqueeze(-1) * ray_d)[:,:,2]
            depth_list.append(depth)
            rgb_list.append(rgb)

        depths = torch.cat(depth_list, dim=1).view(imgH, imgW) # H W
        depths = depths * batch['scale_mat'][0][0, 0]  # scale back
        rgbs = torch.cat(rgb_list, dim=1).view(imgH, imgW,-1)


        depths = depths.cpu().numpy()
        rgbs = rgbs.cpu().numpy()
        rgbs = (rgbs.astype(np.float32) * 255).astype(np.uint8)
        depth_save = ((depths / np.max(depths)).astype(np.float32) * 255).astype(np.uint8)
        Image.fromarray(depth_save).save(os.path.join(self.args.out_dir, scan_name, "depth", "%s.png"%ref_view))
        Image.fromarray(rgbs).save(os.path.join(self.args.out_dir, "rgb", scan_name, "%s.jpg"%ref_view))

        extrinsic_np = batch['extrinsic_render_view'][0].cpu().numpy()

        np.save(os.path.join(self.args.out_dir, "depth", scan_name, "%s.npy"%ref_view), 
                {"depth": depths, "extrinsic":extrinsic_np, "intrinsic": batch['intrinsic_render_view'][0].cpu().numpy()})