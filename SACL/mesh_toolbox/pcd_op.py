import copy
import torch
import numpy as np
import open3d as o3d
from .geom_op import get_param_from_rigid_matrix

# pcd io ops
def read_point_cloud(pcd_path):
    return o3d.io.read_point_cloud(pcd_path)

def write_point_cloud(pcd, pcd_path):
    return o3d.io.write_point_cloud(pcd_path, pcd)

# pcd gen ops
def get_pcd_from_mesh(mesh, compute_normal=True):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh.vertices))
    if(compute_normal):
        pcd.estimate_normals()
    return pcd

# pcd registration ops
def icp_pcd_registration(src_pcd,
                         dst_pcd,
                         threshold=5,
                         max_its=2000,
                         center_x=0,
                         center_y=0,
                         center_z=0,
                         center_align=True,
                         ret_trans_matrix=False,
                         ret_trans_param=False,
                         use_rad=True):
    regis_src_pcd = copy.deepcopy(src_pcd)
    regis_dst_pcd = copy.deepcopy(dst_pcd)
    center_translation = np.array([0, 0, 0])
    # center align pre-process
    if(center_align):
        regis_src_aabb = regis_src_pcd.get_axis_aligned_bounding_box()
        regis_dst_aabb = regis_dst_pcd.get_axis_aligned_bounding_box()
        center_translation = np.array(regis_dst_aabb.get_center()) - np.array(regis_src_aabb.get_center())
        regis_src_pcd.translate(center_translation)
    # registration
    regis_trans = o3d.registration.registration_icp(
        regis_src_pcd, regis_dst_pcd, threshold, np.eye(N=4),
        o3d.registration.TransformationEstimationPointToPlane(),
        o3d.registration.ICPConvergenceCriteria(max_iteration = max_its)
        )
    regis_matrix = regis_trans.transformation.copy()
    # center aligh post-process
    if(center_align):
        regis_matrix[:3, 3] += regis_matrix[:3, :3]@center_translation
    regis_pcd = copy.deepcopy(src_pcd)
    regis_pcd.transform(regis_matrix)
    if(ret_trans_param):
        regis_param = get_param_from_rigid_matrix(regis_matrix,
                                                  center_x=center_x,
                                                  center_y=center_y,
                                                  center_z=center_z,
                                                  use_rad=use_rad)
        return regis_pcd, regis_param
    if(ret_trans_matrix):
        return regis_pcd, regis_matrix
    else:
        return regis_pcd