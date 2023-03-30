import numpy as np
import os, glob, argparse
from plyfile import PlyData, PlyElement
import cv2
from PIL import Image

# read an image
def read_img(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# poject the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3]+1e-6)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres, geo_depth_thres):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)

    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

    
def filter_depth(args, scan):
    # the pair file
    pair_file = os.path.join(args.dataset_dir, "pair.txt")
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    img_path = os.path.join(args.root_dir, "rgb", scan)
    if not args.full_fusion:
        views = [i for i in range(args.n_view)]
    else:
        views = list(range(nviews))

    if not args.full_fusion:
        # for each reference view and the corresponding source views
        for ref_view in views:
            src_views = views[:]
            src_views.pop(ref_view)
            ref_img = read_img(os.path.join(img_path, '{:0>8}.jpg'.format(ref_view)))
            data_tmp = np.load(os.path.join(args.root_dir, "depth", scan, "{:0>8}.npy".format(ref_view)), allow_pickle=True).item()
            # load the estimated depth of the reference view
            ref_depth_est = data_tmp['depth']
            ref_intrinsics = data_tmp['intrinsic']#[:3,:3]
            ref_extrinsics = data_tmp['extrinsic']
            all_srcview_depth_ests = []

            # compute the geometric mask
            geo_mask_sum = 0
            for src_view in src_views:
                data_tmp = np.load(os.path.join(args.root_dir, "depth", scan, "{:0>8}.npy".format(src_view)), allow_pickle=True).item()
                src_depth_est = data_tmp['depth']
                src_intrinsics = data_tmp['intrinsic']#[:3,:3]
                src_extrinsics = data_tmp['extrinsic']

                geo_mask, depth_reprojected, _, _ = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                    src_depth_est,
                                                    src_intrinsics, src_extrinsics,
                                                    args.geo_pixel_thres, 
                                                    args.geo_depth_thres)
                geo_mask_sum += geo_mask.astype(np.int32)
                all_srcview_depth_ests.append(depth_reprojected)

            depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
            geo_mask = geo_mask_sum >= args.geo_mask_thres


            os.makedirs(os.path.join(args.root_dir, scan, "mask"), exist_ok=True)
            save_mask(os.path.join(args.root_dir, scan, "mask/{:0>8}.png".format(ref_view)), geo_mask)
        

            print("processing {}, ref-view{:0>2}, geo_mask:{:3f}".format(scan, ref_view,
                                                    geo_mask.mean()))


            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

            valid_points = geo_mask
            # print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

            color = ref_img[valid_points]
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                np.vstack((xyz_ref, np.ones_like(x))))[:3]
            vertexs.append(xyz_world.transpose((1, 0)))
            vertex_colors.append((color * 255).astype(np.uint8))

    else:
        for ref_view, src_views in pair_data:
            ref_img = read_img(os.path.join(img_path, '{:0>8}.jpg'.format(ref_view)))
            data_tmp = np.load(os.path.join(args.root_dir, "depth", scan, "{:0>8}.npy".format(ref_view)), allow_pickle=True).item()
            # load the estimated depth of the reference view
            ref_depth_est = data_tmp['depth']
            ref_intrinsics = data_tmp['intrinsic']#[:3,:3]
            ref_extrinsics = data_tmp['extrinsic']
            all_srcview_depth_ests = []

            # compute the geometric mask
            geo_mask_sum = 0
            for src_view in src_views:
                data_tmp = np.load(os.path.join(args.root_dir, "depth", scan, "{:0>8}.npy".format(src_view)), allow_pickle=True).item()
                src_depth_est = data_tmp['depth']
                src_intrinsics = data_tmp['intrinsic']#[:3,:3]
                src_extrinsics = data_tmp['extrinsic']

                geo_mask, depth_reprojected, _, _ = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                    src_depth_est,
                                                    src_intrinsics, src_extrinsics,
                                                    args.geo_pixel_thres, 
                                                    args.geo_depth_thres)
                geo_mask_sum += geo_mask.astype(np.int32)
                all_srcview_depth_ests.append(depth_reprojected)

            depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
            geo_mask = geo_mask_sum >= args.geo_mask_thres

            os.makedirs(os.path.join(args.root_dir, scan, "mask"), exist_ok=True)
            save_mask(os.path.join(args.root_dir, scan, "mask/{:0>8}.png".format(ref_view)), geo_mask)
        

            print("processing {}, ref-view{:0>2}, geo_mask:{:3f}".format(scan, ref_view,
                                                    geo_mask.mean()))


            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

            valid_points = geo_mask
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

            color = ref_img[valid_points]
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                np.vstack((xyz_ref, np.ones_like(x))))[:3]
            vertexs.append(xyz_world.transpose((1, 0)))
            vertex_colors.append((color * 255).astype(np.uint8))


    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    plyfilename = os.path.join(args.root_dir, "pcd", "{}.ply".format(scan))
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == "__main__":
    # -------------------------------- args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default="DTU",
        help='dataset name')
    parser.add_argument('--dataset_dir', dest='dataset_dir', type=str,
        help='directory of dataset')
    parser.add_argument('--root_dir', dest='root_dir', type=str,
        help='directory of srdf volumes')
    parser.add_argument('--n_view', dest='n_view', type=int, default=3)
    parser.add_argument('--geo_pixel_thres', type=float, default=1, help='pixel threshold for geometric consistency filtering')
    parser.add_argument('--geo_depth_thres', type=float, default=0.01, help='depth threshold for geometric consistency filtering')
    parser.add_argument('--geo_mask_thres', type=int, default=2, help='number of consistent views for geometric consistency filtering')
    parser.add_argument('--set', dest='set', type=int, default=0)
    parser.add_argument('--full_fusion', dest='full_fusion', action="store_true",
        help='fuse all the depth maps')
    args = parser.parse_args()

    scans = os.listdir(args.root_dir)
    scans = [i for i in scans if i[:4] == 'scan']
    print("found scans:", scans)

    os.makedirs(os.path.join(args.root_dir, "pcd"), exist_ok=True)

    for scan in scans:
        filter_depth(args, scan)
