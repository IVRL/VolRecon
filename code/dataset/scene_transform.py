import numpy as np
import torch

def rigid_transform(xyz, transform):
    """Applies a rigid transform (c2w) to an (N, 3) pointcloud.
    """
    device = xyz.device
    xyz_h = torch.cat([xyz, torch.ones((len(xyz), 1)).to(device)], dim=1)  # (N, 4)
    xyz_t_h = (transform @ xyz_h.T).T  # * checked: the same with the below

    return xyz_t_h[:, :3]


def get_view_frustum(min_depth, max_depth, size, cam_intr, c2w):
    """Get corners of 3D camera view frustum of depth image
    -----------------------------
    Params:
        min_depth, max_depth in mm, i.e., 425.0, 905.0
        size: image size, i.e., 512, 640
        cam_intr: camera intrinsics
        c2w: extrinsics
    
    Return:
        8 corner point in the camera frustum in the world space
    """
    device = cam_intr.device
    im_h, im_w = size
    im_h = int(im_h)
    im_w = int(im_w)
    
    # get point 3D coordinate in the camera coordinate
    # pixel (0~im_w, 0~im_h) -> camera
    view_frust_pts = torch.stack([
        (torch.tensor([0, 0, im_w, im_w, 0, 0, im_w, im_w]).to(device) - cam_intr[0, 2]) * torch.tensor(
            [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(device) /
        cam_intr[0, 0],
        (torch.tensor([0, im_h, 0, im_h, 0, im_h, 0, im_h]).to(device) - cam_intr[1, 2]) * torch.tensor(
            [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(device) /
        cam_intr[1, 1],
        torch.tensor([min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(
            device)
    ])

    view_frust_pts = view_frust_pts.type(torch.float32)
    view_frust_pts = rigid_transform(view_frust_pts.T, c2w).T
    return view_frust_pts


def set_pixel_coords(h, w):
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type(torch.float32)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type(torch.float32)  # [1, H, W]
    ones = torch.ones(1, h, w).type(torch.float32)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    return pixel_coords


def get_boundingbox(img_hw, intrinsics, extrinsics, near_fars):
    """
    # get the minimum bounding box of all visual hulls`
    :param img_hw:
    :param intrinsics:
    :param extrinsics:
    :param near_fars:

    :return:
    ------------------
        center of 8 points (axis-wise)
        radius: half length of max length
        bnds: x,y,z bounding box
    """

    bnds = torch.zeros((3, 2))
    bnds[:, 0] = np.inf
    bnds[:, 1] = -np.inf

    if isinstance(intrinsics, list):
        num = len(intrinsics)
    else:
        num = intrinsics.shape[0]

    for i in range(num):
        if not isinstance(intrinsics[i], torch.Tensor):
            cam_intr = torch.tensor(intrinsics[i])
            w2c = torch.tensor(extrinsics[i])
            c2w = torch.inverse(w2c)
        else:
            cam_intr = intrinsics[i]
            w2c = extrinsics[i]
            c2w = torch.inverse(w2c)
        min_depth, max_depth = near_fars[i][0], near_fars[i][1]
        view_frust_pts = get_view_frustum(min_depth, max_depth, img_hw, cam_intr, c2w)
        bnds[:, 0] = torch.min(bnds[:, 0], torch.min(view_frust_pts, dim=1)[0])
        bnds[:, 1] = torch.max(bnds[:, 1], torch.max(view_frust_pts, dim=1)[0])

    center = torch.tensor(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2,
                           (bnds[2, 1] + bnds[2, 0]) / 2))

    lengths = bnds[:, 1] - bnds[:, 0]

    max_length, _ = torch.max(lengths, dim=0)
    radius = max_length / 2

    return center, radius, bnds
