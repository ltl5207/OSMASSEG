import os
import copy
import torch
import numpy as np
import shutil
import open3d as o3d
import SimpleITK as itk
from scipy.spatial import distance
from skimage.measure import marching_cubes
from .geom_op import get_param_from_rigid_matrix
from .common import make_parent_dir, common_json_load, common_json_dump

# mesh io ops
def read_triangle_mesh(mesh_path):
    return o3d.io.read_triangle_mesh(mesh_path)

def write_triangle_mesh(mesh, mesh_path):
    return o3d.io.write_triangle_mesh(mesh_path, mesh)

def read_triangle_mesh_pv(mesh_path):
    import pyvista as pv
    return pv.read(mesh_path)

# mesh vertices access ops
def get_mesh_vertices(mesh):
    return np.array(mesh.vertices)

def set_mesh_vertices(mesh, vertices):
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

def get_pv_mesh_vertices_arrayND(pv_mesh):
    pv_mesh_verts = copy.copy(np.array(pv_mesh.points))
    return pv_mesh_verts

def set_pv_mesh_vertices_arrayND(pv_mesh, verts_array):
    pv_mesh.points = copy.copy(verts_array)
    return pv_mesh

def get_mesh_vertex_norm_compute_o3dmesh(mesh):
    mesh.compute_vertex_normals()
    return mesh

# mesh gen ops
def get_mesh_from_vol(cube,
                      mesh_colors=[0,255,0],
                      step=1,
                      filter_smooth_taubin_flag=True,
                      filter_smooth_taubin_iter=50,
                      filter_connected_component_flag=True,
                      filter_num_triangle_thresh=500,
                      use_colors=False):
    # cube: 3-dimensional array
    march_cube=1-cube
    # mesh gen
    mesh=o3d.geometry.TriangleMesh()
    if(np.sum(cube)==0):
        print("get_mesh_from_vol: sum(cube)=0")
        return mesh
    mesh_verts,mesh_faces,mesh_norms,_=marching_cubes(march_cube,step_size=step,allow_degenerate=False,method="lewiner")
    mesh.vertices=o3d.utility.Vector3dVector(mesh_verts)
    mesh.triangles=o3d.utility.Vector3iVector(mesh_faces)
    mesh.vertex_normals=o3d.utility.Vector3dVector(mesh_norms)
    mesh.compute_vertex_normals()
    # mesh connect component
    if(filter_connected_component_flag):
        triangle_clusterids,cluster_triangle_nums,_=mesh.cluster_connected_triangles()
        triangle_clusterids=np.array(triangle_clusterids)
        cluster_triangle_nums=np.array(cluster_triangle_nums)
        remove_tids=cluster_triangle_nums[triangle_clusterids]<filter_num_triangle_thresh
        mesh.remove_triangles_by_mask(remove_tids)
    # mesh smooth
    if(filter_smooth_taubin_flag):
        mesh=mesh.filter_smooth_taubin(number_of_iterations=filter_smooth_taubin_iter)
    mesh.compute_vertex_normals()
    vertex_num=np.array(mesh.vertices).shape[0]
    if(use_colors):
        mesh_colors=np.repeat(np.array(mesh_colors)[:,np.newaxis],repeats=vertex_num,axis=1).transpose([1,0])/255
        mesh.vertex_colors=o3d.utility.Vector3dVector(mesh_colors)
    nan_mask=np.isnan(np.sum(np.array(mesh.vertices),axis=1))
    if(np.sum(nan_mask)!=0):
        mesh.remove_vertices_by_mask(nan_mask)
    return mesh

# registration
def icp_mesh_registration(src_mesh,
                          dst_mesh,
                          threshold=5,
                          max_its=2000,
                          center_x=0,
                          center_y=0,
                          center_z=0,
                          center_align=True,
                          ret_trans_matrix=False,
                          ret_trans_param=False,
                          use_rad=True):
    from .pcd_op import get_pcd_from_mesh, icp_pcd_registration
    src_pcd = get_pcd_from_mesh(src_mesh)
    dst_pcd = get_pcd_from_mesh(dst_mesh)
    regis_pcd, regis_matrix = icp_pcd_registration(src_pcd=src_pcd,
                                                   dst_pcd=dst_pcd,
                                                   center_x=center_x,
                                                   center_y=center_y,
                                                   center_z=center_z,
                                                   threshold=threshold,
                                                   max_its=max_its,
                                                   center_align=center_align,
                                                   ret_trans_matrix=True,
                                                   use_rad=use_rad)
    regis_mesh = o3d.geometry.TriangleMesh()
    regis_mesh.vertices = regis_pcd.points
    regis_mesh.triangles = src_mesh.triangles
    regis_mesh.compute_vertex_normals()
    if(ret_trans_param):
        regis_param = get_param_from_rigid_matrix(regis_matrix,
                                                  center_x=center_x,
                                                  center_y=center_y,
                                                  center_z=center_z,
                                                  use_rad=use_rad,
                                                  )
        return regis_mesh, regis_param
    elif(ret_trans_matrix):
        return regis_mesh, regis_matrix
    else:
        return regis_mesh

def cpd_mesh_nonrigid_registration_probreg(
    src_mesh,
    dst_mesh,
    use_cuda=False
):
    from probreg import cpd
    
    # init cuda setting
    if use_cuda:
        import cupy as cp
        to_cpu = cp.asnumpy
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else:
        cp = np
        to_cpu = lambda x: x
    
    # init verts
    src_mesh_verts = get_mesh_vertices(src_mesh)
    dst_mesh_verts = get_mesh_vertices(dst_mesh)
    src_cp_verts = cp.asarray(src_mesh_verts, dtype=cp.float32)
    dst_cp_verts = cp.asarray(dst_mesh_verts, dtype=cp.float32)
    # cpd nonrigid regis
    nonrigid_cpd = cpd.NonRigidCPD(src_cp_verts, use_cuda=use_cuda)
    regis_param, _, _ = nonrigid_cpd.registration(dst_cp_verts)
    regis_cp_verts = regis_param.transform(src_cp_verts)
    regis_mesh_verts = to_cpu(regis_cp_verts)
    # gen registered mesh
    regis_mesh = copy.copy(src_mesh)
    regis_mesh = set_mesh_vertices(regis_mesh, regis_mesh_verts)
    regis_mesh = get_mesh_vertex_norm_compute_o3dmesh(regis_mesh)
    return regis_mesh

def zoom_mesh(mesh, zoom_ratio=[1.0, 1.0, 1.0]):
    mesh_vertices = np.array(mesh.vertices)
    zoom_mesh = copy.copy(mesh)
    zoom_mesh_vertices = mesh_vertices*(np.array(zoom_ratio)[np.newaxis, :])
    zoom_mesh.vertices = o3d.utility.Vector3dVector(zoom_mesh_vertices)
    return zoom_mesh

# mesh metric
def cal_msd_forward(
    mesh_src,
    mesh_dst,
    scale=1.0,
    print_msg_flag=True
):
    """
    notes:
        calculate msd from 'mesh_src' to 'mesh_dst'.

        [Important] 

        This implementation use vertice of 'mesh_src' to find the closest vertice of 'mesh_dst', 
        thus is a discrete format of MSD calculation, 
        'Not' use continues point from 'mesh_src' to 
        calculate the closest distance to the triangle piece of the 'mesh_dst'.
    """
    if(print_msg_flag):
        print("Calculate Msd Forward")
    mesh_src_pts = np.array(mesh_src.vertices)
    mesh_dst_pts = np.array(mesh_dst.vertices)
    src_pts_tensor = torch.FloatTensor(mesh_src_pts)[:,np.newaxis,:]
    dst_pts_tensor = torch.FloatTensor(mesh_dst_pts)[np.newaxis,:,:]
    dist_matrix = torch.sqrt(torch.sum((src_pts_tensor-dst_pts_tensor)**2, dim=2))
    dist_min = torch.min(dist_matrix, dim=1).values
    th = 1
    dist_min = (dist_min >= th) * th + (dist_min < th) * dist_min
    msd = torch.mean(dist_min).numpy()*scale
    return msd

def cal_msd_backward(
    mesh_src,
    mesh_dst,
    scale=1.0,
    print_msg_flag=True
):
    """
    notes:
        calculate msd from 'mesh_dst' to 'mesh_src'.

        [Important] 
        
        This implementation use vertice of 'mesh_dst' to find the closest vertice of 'mesh_src', 
        thus is a discrete format of MSD calculation, 
        'Not' use continues point from 'mesh_dst' to 
        calculate the closest distance to the triangle piece of the 'mesh_src'.
    """
    if(print_msg_flag):
        print("Calculate Msd Backward")
    cur_mesh_src = mesh_src
    cur_mesh_dst = mesh_dst
    msd = cal_msd_forward(
        mesh_src=cur_mesh_dst,
        mesh_dst=cur_mesh_src,
        scale=scale,
        print_msg_flag=False
    )
    return msd

def cal_msd_symmetric(
    mesh_a,
    mesh_b,
    scale=1.0,
    print_msg_flag=True
):
    """
    notes:
        calculate msd symmetrically, average the result of 'cal_msd_forward' and 'cal_msd_backward'

        [Important] 

        This implementation use vertice of 'mesh_a/mesh_b' to find the closest vertice of 'mesh_b/mesh_a', 
        thus is a discrete format of MSD calculation, 
        'Not' use continues point from 'mesh_a/mesh_b' to 
        calculate the closest distance to the triangle piece of the 'mesh_b/mesh_a'.
    """
    if(print_msg_flag):
        print("Calculate Msd Symmetric")
    # forward
    msd_forward = cal_msd_forward(
        mesh_src=mesh_a,
        mesh_dst=mesh_b,
        scale=scale,
        print_msg_flag=False
    )
    # backward
    msd_backward = cal_msd_backward(
        mesh_src=mesh_a,
        mesh_dst=mesh_b,
        scale=scale,
        print_msg_flag=False
    )
    msd_symmetric = (msd_forward + msd_backward)/2
    return msd_symmetric

def get_pv_mesh_forward_msd_vec_arrayND(src_mesh,
                                    ref_mesh,
                                    scale=1.0):
    # src_mesh and ref_mesh are pv mesh object
    src_mesh_pts = src_mesh.points
    ref_mesh_pts = ref_mesh.points
    dis_matrix = distance.cdist(src_mesh_pts, ref_mesh_pts, metric="euclidean")
    dis_matrix[np.isnan(dis_matrix)] = 100000
    msd_vector = np.min(dis_matrix, axis=1)*scale
    return msd_vector

def get_pv_mesh_symmetric_msd_byregismesh_arrayND(src_mesh,
                                                  ref_mesh,
                                                  regis_src_mesh,
                                                  scale=1.0,
                                                  dist_matrix_nan_value=100000):
    src_mesh_pts = get_pv_mesh_vertices_arrayND(src_mesh)
    dst_mesh_pts = get_pv_mesh_vertices_arrayND(ref_mesh)
    regis_mesh_pts = get_pv_mesh_vertices_arrayND(regis_src_mesh)
    # init
    input_src_mesh = src_mesh
    input_ref_mesh = ref_mesh
    # forward msd vec
    fwd_msd_vec = get_pv_mesh_forward_msd_vec_arrayND(src_mesh=input_src_mesh, ref_mesh=input_ref_mesh, scale=scale)
    # backward msd vec
    bkd_msd_vec = get_pv_mesh_forward_msd_vec_arrayND(src_mesh=input_ref_mesh, ref_mesh=input_src_mesh, scale=scale)
    # find correspondence
    dist_matrix = get_dist_matrix_arrayND(src_pts=regis_mesh_pts, dst_pts=dst_mesh_pts, set_nan_value=dist_matrix_nan_value)
    corres_idxs = np.argmin(dist_matrix, axis=1)
    corres_bkd_msd_vec = bkd_msd_vec[corres_idxs]
    # print(f"test shape fwd_msd_vec:{fwd_msd_vec.shape} bkd_msd_vec:{bkd_msd_vec.shape} dist_matrix:{dist_matrix.shape} corres_idxs:{corres_idxs.shape} corres_bkd_msd_vec:{corres_bkd_msd_vec.shape}")
    # symmetric msd vec
    sym_msd_vec = (fwd_msd_vec + corres_bkd_msd_vec)/2
    return sym_msd_vec

def get_pv_mesh_corres_msd_byregismesh_arrayND(src_mesh,
                                                  ref_mesh,
                                                  regis_src_mesh,
                                                  scale=1.0,
                                                  dist_matrix_nan_value=100000):
    """
    Calculate Msd based on the correspondence found between regis_src_mesh and ref_mssh
    """
    src_mesh_pts = get_pv_mesh_vertices_arrayND(src_mesh)
    dst_mesh_pts = get_pv_mesh_vertices_arrayND(ref_mesh)
    regis_mesh_pts = get_pv_mesh_vertices_arrayND(regis_src_mesh)
    # init
    input_src_mesh = src_mesh
    input_ref_mesh = ref_mesh
    # find correspondence
    dist_matrix = get_dist_matrix_arrayND(src_pts=regis_mesh_pts, dst_pts=dst_mesh_pts, set_nan_value=dist_matrix_nan_value)
    corres_idxs = np.argmin(dist_matrix, axis=1)
    # calculate correspond distance
    corres_dst_mesh_pts = dst_mesh_pts[corres_idxs, :]
    # print(f"test shape src_mesh_pts.shape:{src_mesh_pts.shape} dst_mesh_pts.shape:{dst_mesh_pts.shape}")
    # print(f"test shape corres_idxs.shape:{corres_idxs.shape} corres_dst_mesh_pts.shape:{corres_dst_mesh_pts.shape}")
    corres_dist = np.sqrt(np.sum((src_mesh_pts - corres_dst_mesh_pts)**2, axis=1))
    # print(f"test shape corres_dist.shape:{corres_dist.shape}")
    corres_msd_vec = corres_dist*scale
    return corres_msd_vec

def get_dist_matrix_arrayND(src_pts, dst_pts, set_nan_value=100000):
    """
    src_pts: array, [N, 3]
    dst_pts: array, [N, 3]
    """
    from scipy.spatial import distance
    input_src_pts = np.array(copy.copy(src_pts))
    input_dst_pts = np.array(copy.copy(dst_pts))
    dist_matrix = distance.cdist(input_src_pts, input_dst_pts, metric="euclidean")
    dist_matrix[np.isnan(dist_matrix)] = set_nan_value
    return dist_matrix

def get_msd_mesh(src_mesh,ref_mesh,norm_range=5):
    msd_mesh=copy.copy(src_mesh)
    ref_vertices=np.array(ref_mesh.vertices)
    src_vertices=np.array(src_mesh.vertices)
    dis_matrix=distance.cdist(src_vertices,ref_vertices,metric="euclidean")
    dis_matrix[np.isnan(dis_matrix)]=100000
    min_dis_vector=np.min(dis_matrix,axis=1)
    norm_dis_vector=np.clip(min_dis_vector,0,norm_range)/norm_range
    msd_colors=np.array([0,255,0])[:,np.newaxis]*(1-norm_dis_vector)+np.array([255,0,0])[:,np.newaxis]*norm_dis_vector
    msd_colors=np.clip(msd_colors,0,255).transpose([1,0])/255
    msd_mesh.vertex_colors=o3d.utility.Vector3dVector(msd_colors)
    return msd_mesh

class MeshDrawer:
    def __init__(self,
                 camera_position_path,
                 show_windows=False,
                 window_size=[512,512],
                 light_system_type="preset_1",
                 ):
        """
        available light system: defualt, preset_1
        """
        import pyvista as pv
        self.camera_position_path = camera_position_path
        self.show_windows = show_windows
        self.window_size = window_size
        # plotter setting
        self.plotter_window_size = [self.window_size[1], self.window_size[0]]
        # set camera position
        make_parent_dir(self.camera_position_path)
        # camera position button
        self.show_camera_position_save_button_flag = False
        self.camera_position_save_dir = None
        self.camera_position_save_perfix = None
        self.camera_position_save_idx = -1
        self.camera_position_save_button_start_pos = None
        self.camera_position_save_button_size = None
        self.camera_position_save_print_info_flag = False
        
        
        
        # initialize light system
        self.light_system_type = light_system_type
        default_plotter = pv.Plotter()
        default_light_list = default_plotter.renderer.lights
        # default
        if(self.light_system_type == "default"):
            self.light_list = default_light_list
        # preset_1
        if(self.light_system_type == "preset_1"):
            preset_1_light_list = copy.copy(default_light_list)
            for preset_light in preset_1_light_list:
                preset_light.intensity -= 0.13
                preset_light.intensity = np.clip(preset_light.intensity, 0, 100)
            downside_light = pv.Light(position=(0, -10, 0), focal_point=[0, 10, 0], intensity=0.4, light_type='camera light')
            frontside_light = pv.Light(position=(0, 0, 10), focal_point=[0, 0, -10], intensity=0.3, light_type='camera light')
            leftside_light = pv.Light(position=(-10, 0, 0), focal_point=[10, 0, 0], intensity=0.2, light_type='camera light')
            # rightside_light = pv.Light(position=(10, 0, 0), focal_point=[-10, 0, 0], intensity=0.1, light_type='camera light')
            preset_1_light_list.append(downside_light)
            preset_1_light_list.append(frontside_light)
            preset_1_light_list.append(leftside_light)
            # preset_1_light_list.append(rightside_light)
            self.light_list = preset_1_light_list
        
        # draw mesh by stages
        self.draw_stage_mesh_list = []
        self.draw_stage_mesh_color_list = []
        self.draw_stage_kwargsdict_list = []
    
    def load_camera_position(self, camera_position_path):
        if(os.path.exists(camera_position_path)):
            self.plotter.camera_position = common_json_load(camera_position_path)

    def save_camera_position(self, camera_postion_path, print_camera_flag=False):
        common_json_dump(list(self.plotter.camera_position), camera_postion_path)
        if(print_camera_flag):
            print(self.plotter.camera_position)
    
    def get_cur_camera_position(self):
        return list(self.plotter.camera_position)
    
    def del_camera_position(self, camera_position_path):
        os.remove(camera_position_path)
        return True
    
    def activate_camera_positon_save_ability(self):
        self.show_camera_position_save_button_flag = True
        return True

    def deactivate_camera_position_save_ability(self):
        self.show_camera_position_save_button_flag = False
    
    def activate_show_windows_ability(self):
        self.show_windows = True
    
    def deactivate_show_windows_ability(self):
        self.show_windows = False
        
    def set_camera_position_save_param(self,
                                       camera_position_save_dir,
                                       camera_position_save_perfix,
                                       camera_position_save_button_start_pos=(10, 10),
                                       camera_position_save_button_size=50,
                                       camera_position_save_print_info_flag=False):
        self.camera_position_save_dir = camera_position_save_dir
        self.camera_position_save_perfix = camera_position_save_perfix
        self.camera_position_save_button_start_pos = camera_position_save_button_start_pos
        self.camera_position_save_button_size = camera_position_save_button_size
        self.camera_position_save_print_info_flag = camera_position_save_print_info_flag
    
    def get_cur_camera_position_save_path(self):
        return f"{self.camera_position_save_dir}/{self.camera_position_save_perfix}_{self.camera_position_save_idx}.json"
    
    def get_prev_camera_position_save_path(self, save_idx):
        return f"{self.camera_position_save_dir}/{self.camera_position_save_perfix}_{save_idx}.json"
    
    def draw_mesh(self,
                  pv_mesh,
                  background_color=[1.0, 1.0, 1.0],
                  mesh_color=None,
                  save_screenshot_path=False,
                  cam_pos_path=None,
                  auto_load_pos_flag=True,
                  **kwargs
                ):
        import pyvista as pv
        
        # init plotter
        self.plotter = pv.Plotter(lighting="none",
                                  off_screen=(not self.show_windows), 
                                  window_size=self.plotter_window_size)
        
        # camera position save func
        def save_camera_position_callback(button_flag):
            if(button_flag==True):
                self.camera_position_save_idx += 1
                if(self.camera_position_save_idx>=0):
                    cur_camera_position_save_path = self.get_cur_camera_position_save_path()
                    self.save_camera_position(camera_postion_path=cur_camera_position_save_path)
                    print(f"saved camera position at {cur_camera_position_save_path}")
                    if(self.camera_position_save_print_info_flag):
                        print(f"saved camera position:{self.get_cur_camera_position()}")
                else:
                    print(f"current camera_position_save_idx={self.camera_position_save_idx}<0, ignore")
                return True
            return False
        
        def del_camera_position_callback(button_flag):
            if(button_flag==True):
                if(self.camera_position_save_idx>=0):
                    cur_camera_position_save_path = self.get_cur_camera_position_save_path()
                    self.camera_position_save_idx -= 1
                    self.del_camera_position(cur_camera_position_save_path)
                    print(f"delete camera position at {cur_camera_position_save_path}")
                    return True
                else:
                    print(f"current camera_position_save_idx={self.camera_position_save_idx}<0, ignore")
            return False
                        
        if(self.show_camera_position_save_button_flag):
            # save call back
            save_button_start_pos = [
                self.camera_position_save_button_start_pos[0],
                self.camera_position_save_button_start_pos[1] + self.camera_position_save_button_size*11//10
            ]
            self.plotter.add_checkbox_button_widget(
                callback=save_camera_position_callback,
                value=False,
                color_on="blue",
                color_off="grey",
                position=save_button_start_pos,
                size=self.camera_position_save_button_size
            )
            # del call back
            del_button_start_pos = self.camera_position_save_button_start_pos
            self.plotter.add_checkbox_button_widget(
                callback=del_camera_position_callback,
                value=False,
                color_on="red",
                color_off="grey",
                position=del_button_start_pos,
                size=self.camera_position_save_button_size
            )
            
        # configure light system
        for preset_light in self.light_list:
            self.plotter.add_light(preset_light)
        
        if(cam_pos_path is not None):
            self.load_camera_position(cam_pos_path)
        if(auto_load_pos_flag):
            self.load_camera_position(self.camera_position_path)
        self.plotter.set_background(color=background_color)
        self.plotter.add_mesh(
            pv_mesh,
            color=mesh_color,
            **kwargs
        )
        make_parent_dir(save_screenshot_path)
        self.plotter.show(screenshot=save_screenshot_path)
        if(auto_load_pos_flag and self.show_windows):
            self.save_camera_position(self.camera_position_path)
    
    def draw_mesh_list(self,
                  pv_mesh_list,
                  mesh_color_list=None,
                  background_color=[1.0, 1.0, 1.0],
                  save_screenshot_path=False,
                  **kwargs
                ):
        import pyvista as pv
        # init plotter
        self.plotter = pv.Plotter(lighting="none",
                                  off_screen=(not self.show_windows), 
                                  window_size=self.plotter_window_size)
        # configure light system
        for preset_light in self.light_list:
            self.plotter.add_light(preset_light)
        
        self.load_camera_position(self.camera_position_path)
        self.plotter.set_background(color=background_color)
        for pv_mesh, mesh_color in zip(pv_mesh_list, mesh_color_list):
            self.plotter.add_mesh(
                pv_mesh,
                color=mesh_color,
                **kwargs
            )
        make_parent_dir(save_screenshot_path)
        self.plotter.show(screenshot=save_screenshot_path)
        self.save_camera_position(self.camera_position_path)
    
    def add_mesh_stage(self,
                    pv_mesh,
                    mesh_color,
                    **kwargs
                ):
        self.draw_stage_mesh_list.append(pv_mesh)
        self.draw_stage_mesh_color_list.append(mesh_color)
        self.draw_stage_kwargsdict_list.append(kwargs)
    
    def clean_mesh_stage(self):
        self.draw_stage_mesh_list = []
        self.draw_stage_mesh_color_list = []
        self.draw_stage_kwargsdict_list = []
    
    def draw_mesh_stage(self,
                        background_color=[1.0, 1.0, 1.0],
                        save_screenshot_path=False,
                        auto_load_pos_flag=True
                        ):
        
        import pyvista as pv
        # init plotter
        self.plotter = pv.Plotter(lighting="none",
                                  off_screen=(not self.show_windows), 
                                  window_size=self.plotter_window_size)
        # configure light system
        for preset_light in self.light_list:
            self.plotter.add_light(preset_light)
        
        # load camera position
        if(auto_load_pos_flag):
            self.load_camera_position(self.camera_position_path)
        # draw mesh
        self.plotter.set_background(color=background_color)
        for pv_mesh, mesh_color, kwargs_dict in zip(self.draw_stage_mesh_list, self.draw_stage_mesh_color_list, \
            self.draw_stage_kwargsdict_list):
            if("save_screenshot_path" in kwargs_dict):
                kwargs_dict.remove("save_screenshot_path")
            self.plotter.add_mesh(
                pv_mesh,
                color=mesh_color,
                **kwargs_dict
            )
        make_parent_dir(save_screenshot_path)
        self.plotter.show(screenshot=save_screenshot_path)
        # save camera position
        if(auto_load_pos_flag and self.show_windows):
            self.save_camera_position(self.camera_position_path)
        
        