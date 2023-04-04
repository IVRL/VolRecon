import torch
import cv2 as cv
import numpy as np
import os
import logging
from einops import repeat
from .scene_transform import get_boundingbox


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class GeneralFit:
    def __init__(self, root_dir, scan_id, n_views=5,
                 img_wh=[800, 600], clip_wh=[0, 0], 
                 N_rays=512):
        super(GeneralFit, self).__init__()
        logging.info('Load data: Begin')

        self.root_dir = root_dir
        self.scan_id = scan_id
        self.offset_dist = 0.025 # 25mm, assume the metric is meter
        self.n_views = n_views 
        self.metas = self.build_list()
        self.num_img = len(self.metas)
        
        self.test_img_idx = list(self.metas)

        self.data_dir = os.path.join(self.root_dir, self.scan_id)

        self.img_wh = img_wh
        self.clip_wh = clip_wh

        if len(self.clip_wh) == 2:
            self.clip_wh = self.clip_wh + self.clip_wh

        self.N_rays = N_rays
        
        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])
        self.partial_vol_origin = torch.Tensor([-1., -1., -1.])

        self.img_W, self.img_H = self.img_wh
        h_line = (np.linspace(0,self.img_H-1,self.img_H))*2/(self.img_H-1) - 1
        w_line = (np.linspace(0,self.img_W-1,self.img_W))*2/(self.img_W-1) - 1
        h_mesh, w_mesh = np.meshgrid(h_line, w_line, indexing='ij')
        self.w_mesh_flat = w_mesh.reshape(-1)
        self.h_mesh_flat = h_mesh.reshape(-1)
        self.homo_pixel = np.stack([self.w_mesh_flat, self.h_mesh_flat, np.ones(len(self.h_mesh_flat)), np.ones(len(self.h_mesh_flat))])

        logging.info('Load data: End')


    def build_list(self):
        metas = []
        pair_file = os.path.join(self.root_dir, self.scan_id, "pair.txt")
            
        # read the pair file
        with open(pair_file) as f:
            num_viewpoint = int(f.readline())
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                metas.append((ref_view, src_views))
        # print("dataset", "metas:", len(metas))
        return metas


    def read_cam_file(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        P = intrinsics_ @ extrinsics
        # depth_min & depth_interval: line 11
        near = float(lines[11].split()[0])
        far = float(lines[11].split()[-1])

        return P, near, far


    def load_scene(self, images_list, world_mats_np, ref_w2c):
        all_images = []
        all_intrinsics = []
        all_w2cs = []
        all_w2cs_original = []
        all_render_w2cs = []
        all_render_w2cs_original = []

        for idx in range(len(images_list)):
            image = cv.imread(images_list[idx])
            original_h, original_w, _ = image.shape
            scale_x = self.img_wh[0] / original_w
            scale_y = self.img_wh[1] / original_h
            image = cv.resize(image, (self.img_wh[0], self.img_wh[1])) / 255.

            image = image[self.clip_wh[1]:self.img_wh[1] - self.clip_wh[3],
                    self.clip_wh[0]:self.img_wh[0] - self.clip_wh[2]]
            all_images.append(np.transpose(image[:, :, ::-1], (2, 0, 1)))

            P = world_mats_np[idx]
            P = P[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            w2c = np.linalg.inv(c2w)

            render_c2w = c2w.copy()
            render_c2w[:3,3] += render_c2w[:3,0]*self.offset_dist
            render_w2c = np.linalg.inv(render_c2w)

            intrinsics[:1] *= scale_x
            intrinsics[1:2] *= scale_y

            intrinsics[0, 2] -= self.clip_wh[0]
            intrinsics[1, 2] -= self.clip_wh[1]

            all_intrinsics.append(intrinsics)
            # - transform from world system to ref-camera system
            all_w2cs.append(w2c @ np.linalg.inv(ref_w2c))
            all_render_w2cs.append(render_w2c @ np.linalg.inv(ref_w2c))
            all_w2cs_original.append(w2c)
            all_render_w2cs_original.append(render_w2c)

        all_images = torch.from_numpy(np.stack(all_images)).to(torch.float32)
        all_intrinsics = torch.from_numpy(np.stack(all_intrinsics)).to(torch.float32)
        all_w2cs = torch.from_numpy(np.stack(all_w2cs)).to(torch.float32)
        all_render_w2cs = torch.from_numpy(np.stack(all_render_w2cs)).to(torch.float32)

        return all_images, all_intrinsics, all_w2cs, all_w2cs_original, all_render_w2cs, all_render_w2cs_original


    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()


    def scale_cam_info(self, all_images, all_intrinsics, all_w2cs, scale_mat, all_render_w2cs):
        new_intrinsics = []
        new_w2cs = []
        new_c2ws = []
        new_render_w2cs = []
        new_render_c2ws = []
        for idx in range(len(all_images)):
            intrinsics = all_intrinsics[idx]
            P = intrinsics @ all_w2cs[idx] @ scale_mat
            P = P.cpu().numpy()[:3, :4]

            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            new_intrinsics.append(intrinsics)

            P = intrinsics @ all_render_w2cs[idx] @ scale_mat
            P = P.cpu().numpy()[:3, :4]

            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_render_w2cs.append(w2c)
            new_render_c2ws.append(c2w)

        new_intrinsics, new_w2cs, new_c2ws = \
            np.stack(new_intrinsics), np.stack(new_w2cs), np.stack(new_c2ws)
        new_render_w2cs, new_render_c2ws = np.stack(new_render_w2cs), np.stack(new_render_c2ws)

        new_intrinsics = torch.from_numpy(np.float32(new_intrinsics))
        new_w2cs = torch.from_numpy(np.float32(new_w2cs))
        new_c2ws = torch.from_numpy(np.float32(new_c2ws))
        new_render_w2cs = torch.from_numpy(np.float32(new_render_w2cs))
        new_render_c2ws = torch.from_numpy(np.float32(new_render_c2ws))

        return new_intrinsics, new_w2cs, new_c2ws, new_render_w2cs, new_render_c2ws


    def __len__(self):
        return self.num_img


    def __getitem__(self, idx):
        sample = {}
        ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views-1]
        idx = list(range(self.n_views))
        render_idx = 0
        src_idx = idx

        world_mats_np = []
        images_list = []
        raw_near_fars = []
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.data_dir, 'images/{:0>8}.jpg'.format(vid))
            images_list.append(img_filename)

            proj_mat_filename = os.path.join(self.data_dir, 'cameras/{:0>8}_cam.txt'.format(vid))
            P, near_, far_ = self.read_cam_file(proj_mat_filename)

            raw_near_fars.append(np.array([near_,far_]))
            world_mats_np.append(P)
        raw_near_fars = np.stack(raw_near_fars)
        ref_world_mat = world_mats_np[0]
        ref_w2c = np.linalg.inv(load_K_Rt_from_P(None, ref_world_mat[:3, :4])[1])

        all_images, all_intrinsics, all_w2cs, all_w2cs_original, all_render_w2cs, all_render_w2cs_original = self.load_scene(images_list, 
                                                                                                                    world_mats_np, ref_w2c)

        scale_mat, scale_factor = self.cal_scale_mat(
            img_hw=[self.img_wh[1], self.img_wh[0]],
            intrinsics=all_intrinsics,
            extrinsics=all_w2cs,
            near_fars=raw_near_fars,
            factor=1.1)
        scaled_intrinsics, scaled_w2cs, scaled_c2ws, scaled_render_w2cs, scaled_render_c2ws = self.scale_cam_info(all_images, 
                                                                            all_intrinsics, all_w2cs, scale_mat, all_render_w2cs)

        near_fars = torch.tensor(raw_near_fars[0])
        near_fars = near_fars/scale_mat[0,0]
        near_fars = near_fars.view(1,2)
        sample['near_fars'] = near_fars.float()

        sample['scale_mat'] = torch.from_numpy(scale_mat)
        sample['trans_mat'] = torch.from_numpy(np.linalg.inv(ref_w2c))
        sample['extrinsic_render_view'] = torch.from_numpy(all_render_w2cs_original[render_idx])
        
        sample['w2cs'] = scaled_w2cs  # (V, 4, 4)
        sample['intrinsics'] = scaled_intrinsics[:, :3, :3]  # (V, 3, 3)
        sample['intrinsic_render_view'] = sample['intrinsics'][render_idx]

        sample['ref_img'] = all_images[render_idx]
        sample['source_imgs'] = all_images[src_idx]

        intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = len(sample['w2cs'])).clone()
        intrinsics_pad[:,:3,:3] = sample['intrinsics']
        
        sample['ref_pose']         = (intrinsics_pad @ scaled_render_w2cs)[render_idx]     # 4, 4
        sample['source_poses']     = (intrinsics_pad @ sample['w2cs'])[src_idx]

        # from 0~W to NDC's -1~1
        normalize_matrix = torch.tensor([[1/((self.img_W-1)/2), 0, -1, 0], [0, 1/((self.img_H-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]])

        sample['ref_pose'] = normalize_matrix @ sample['ref_pose']
        sample['source_poses'] = normalize_matrix @ sample['source_poses']

        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose'])
        sample['source_poses_inv'] = torch.inverse(sample['source_poses'])

        sample['ray_o'] = sample['ref_pose_inv'][:3,-1]      # 3

        tmp_ray_d = (sample['ref_pose_inv'] @ self.homo_pixel)[:3] - sample['ray_o'][:,None]

        sample['ray_d'] = tmp_ray_d / torch.norm(tmp_ray_d, dim=0)
        sample['ray_d'] = sample['ray_d'].float()


        cam_ray_d = ((torch.inverse(normalize_matrix @ intrinsics_pad[0])) @ self.homo_pixel)[:3]
        cam_ray_d = cam_ray_d / torch.norm(cam_ray_d, dim=0)
        sample['cam_ray_d'] = cam_ray_d.float()

        sample['meta'] = "%s-%s-%08d"%(self.root_dir.split("/")[-1], self.scan_id, ref_view)

        return sample