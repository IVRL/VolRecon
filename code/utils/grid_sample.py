
import torch.nn.functional as F


def grid_sample_2d(input, grid):
    """
    input: (B L) C H W
    grid: (B L) RN SN 2
    """
    
    N, C, H, W = input.shape
    _, RN, SN, _ = grid.shape
    mask = (grid[..., 0] <= 1.) & \
               (grid[..., 0] >= -1.) & \
               (grid[..., 1] <= 1.) &\
               (grid[..., 1] >= -1.)  # (B L) RN SN
    mask = mask.float()
    output = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')  # (B L) C RN SN
    return output, mask


def grid_sample_3d(input, grid):
    """
    input: B C X Y Z
    grid: B 1 RN SN 3
    """

    B, C, X, Y, Z = input.shape
    _, _, RN, SN, _ = grid.shape
    output = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')  # B C 1 RN SN
    return output.squeeze(2)  # B C RN SN