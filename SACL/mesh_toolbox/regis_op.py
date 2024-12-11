from .dependencies import *
from .common import *
from .image_op import *
from .volume_op import *

# plastimatch registration
def plasimatch_apply_trans(input_image_path,
                           input_trans_path,
                           output_image_path,
                           interpolation_type="nn"
                          ):
    """
    Perform transformation over the input image with the given transformation file path.
    notice: 1.all input images are first convert into the np.float32 type.

    Args:
        input_image_path (str): path of input image.
        input_trans_path (str): path of input trans.
        output_image_path (str): path of output image.
        interpolation_type (str, optional): interpolation type. Defaults to "nn".
    """
    input_image = read_mha_array3D(input_image_path).astype(np.float32)
    write_mha_array3D(input_image, input_image_path)
    os.system(f"plastimatch warp --input {input_image_path} --output-img {output_image_path} --xf {input_trans_path} --interpolation {interpolation_type}")

def plastimatch_regis_rigid(fix_image_path, 
                            mov_image_path,
                            output_image_path=None,
                            output_trans_path=None,
                            fix_mask_path=None,
                            mov_mask_path=None,
                            mov_seg_path=None,
                            output_seg_path=None,
                            max_its=300,
                            metric="mse",
                            convergence_tol=1e-6,
                            default_value=0,
                            transfer_input_image_paths=[],
                            transfer_output_image_paths=[],
                            transfer_interpolation_methods=[],
                            temp_dir="./temp_dir",
                            initialize_flag = False,
                            remove_temp_dir=True
                            ):
    """
    Perform rigid registeration between target image(mov image) to template image(fix image).
    notice: 1.mask with value 1 is the region that consider during the plastimatch registering.
    notice: 2.output transform is of txt format.

    Args:
        fix_image_path (str): path of template image.
        mov_image_path (str): path of target image.
        output_image_path (str): path of output image from registeration.
        output_trans_path (str): path of output trans from registeration, txt format.
        fix_mask_path (str): path of template mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_mask_path (str): path of target mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_seg_path (str): path of target segment, used for segment transfer.
        output_seg_path (str): path of output template segment, transformed from mov_seg by output transformation.
        max_its: max iteration number.
        default_value (float): value to be pad in the region that newly produced by the registeration.
        transfer_input_image_paths (List): paths of other images to be transfered, must be the same length of transfer_output_image_paths.
        transfer_output_image_paths (List): paths of other images that output by warping transformation, must be the same length of transfer_input_image_paths.
        transfer_interpolation_methods (List): methods of interpolation to warp other images.
        temp_dir (str): temp directory to save plastimatch cmd file.
        remove_temp_dir (bool): whether to remove temp_dir after registeration.
    """
    # paths
    os.makedirs(temp_dir,exist_ok=True)
    cmd_file_path = f"{temp_dir}/cmd_file.txt"
    if(output_image_path is None):
        output_image_path = f"{temp_dir}/output_image.nii.gz"
    if(output_trans_path is None):
        output_trans_path = f"{temp_dir}/output_trans.txt"
    make_parent_dir(output_image_path)
    make_parent_dir(output_trans_path)
    # rigid registration
    regis_file = open(cmd_file_path, mode="w")
    # global
    regis_file.write("[GLOBAL]\n")
    regis_file.write(f"fixed={fix_image_path}\n")
    regis_file.write(f"moving={mov_image_path}\n")
    if(fix_mask_path is not None):
        regis_file.write(f"fixed_mask={fix_mask_path}\n")
    if(mov_mask_path is not None):
        regis_file.write(f"moving_mask={mov_mask_path}\n")
    regis_file.write(f"img_out={output_image_path}\n")
    regis_file.write(f"xform_out={output_trans_path}\n")
    regis_file.write("\n")
    # initialize
    if(initialize_flag):
        regis_file.write("[STAGE]\n")
        regis_file.write("xform=align_center_of_gravity\n")
        regis_file.write("\n")
    # stage
    regis_file.write("[STAGE]\n")
    regis_file.write("impl=itk\n")
    regis_file.write("xform=rigid\n")
    regis_file.write("optim=versor\n")
    regis_file.write(f"max_its={max_its}\n")
    regis_file.write(f"metric={metric}\n")
    regis_file.write(f"convergence_tol={convergence_tol}\n")
    regis_file.write("res=1 1 1\n")
    regis_file.write("res_vox_moving=1 1 1\n")
    regis_file.write(f"default_value={default_value}\n")
    regis_file.close()
    os.system(f"plastimatch register {cmd_file_path}") 

    # rigid segment transformation 
    if(mov_seg_path is not None and output_seg_path is not None):
        assert(os.path.exists(mov_seg_path))
        mov_seg = read_mha_array3D(mov_seg_path).astype(np.float32)
        write_mha_array3D(mov_seg, mov_seg_path)
        make_parent_dir(output_seg_path)
        os.system(f"plastimatch warp --input {mov_seg_path} --output-img {output_seg_path} --xf {output_trans_path} --interpolation nn")
    
    # transfer input images
    assert len(transfer_input_image_paths)==len(transfer_output_image_paths)
    if(len(transfer_interpolation_methods)==0):
        transfer_interpolation_methods = ["nn"]*len(transfer_input_image_paths)
    for transfer_input_image_path, transfer_output_image_path, transfer_interpolation_method in \
        zip(transfer_input_image_paths, transfer_output_image_paths, transfer_interpolation_methods):
        plasimatch_apply_trans(
            transfer_input_image_path,
            output_trans_path,
            transfer_output_image_path,
            transfer_interpolation_method
        )
    
    if(remove_temp_dir):
        os.system(f"rm -r {temp_dir}")

def plastimatch_regis_affine(fix_image_path, 
                             mov_image_path,
                             output_image_path=None,
                             output_trans_path=None,
                             fix_mask_path=None,
                             mov_mask_path=None,
                             mov_seg_path=None,
                             output_seg_path=None,
                             metric="mse",
                             max_its=300,
                             grad_tol=1e-3,
                             res_list=[1,1,1],
                             default_value=0,
                             transfer_input_image_paths=[],
                             transfer_output_image_paths=[],
                             transfer_interpolation_methods=[],
                             temp_dir="./temp_dir",
                             remove_temp_dir=True
                             ):
    """
    Perform affine registeration between target image(mov image) to template image(fix image).
    notice: 1.mask with value 1 is the region that consider during the plastimatch registering.
    notice: 2.output transform is of txt format.

    Args:
        fix_image_path (str): path of template image.
        mov_image_path (str): path of target image.
        output_image_path (str): path of output image from registeration.
        output_trans_path (str): path of output trans from registeration, txt format.
        fix_mask_path (str): path of template mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_mask_path (str): path of target mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_seg_path (str): path of target segment, used for segment transfer.
        output_seg_path (str): path of output template segment, transformed from mov_seg by output transformation.
        max_its: max iteration number.
        default_value (float): value to be pad in the region that newly produced by the registeration.
        transfer_input_image_paths (List): paths of other images to be transfered, must be the same length of transfer_output_image_paths.
        transfer_output_image_paths (List): paths of other images that output by warping transformation, must be the same length of transfer_input_image_paths.
        transfer_interpolation_methods (List): methods of interpolation to warp other images.
        temp_dir (str): temp directory to save plastimatch cmd file.
        remove_temp_dir (bool): whether to remove temp_dir after registeration.
    """
    # paths
    os.makedirs(temp_dir,exist_ok=True)
    cmd_file_path = f"{temp_dir}/cmd_file.txt"
    if(output_image_path is None):
        output_image_path = f"{temp_dir}/output_image.nii.gz"
    if(output_trans_path is None):
        output_trans_path = f"{temp_dir}/output_trans.txt"
    make_parent_dir(output_image_path)
    make_parent_dir(output_trans_path)
    # affine registration    
    regis_file=open(cmd_file_path,mode="w")
    # global
    regis_file.write("[GLOBAL]\n")
    regis_file.write(f"fixed={fix_image_path}\n")
    regis_file.write(f"moving={mov_image_path}\n")
    if(fix_mask_path is not None):
        regis_file.write(f"fixed_mask={fix_mask_path}\n")
    if(mov_mask_path is not None):
        regis_file.write(f"moving_mask={mov_mask_path}\n")
    regis_file.write(f"img_out={output_image_path}\n")
    regis_file.write("\n")
    # stage
    regis_file.write("[STAGE]\n")
    regis_file.write("impl=itk\n")
    regis_file.write("xform=affine\n")
    regis_file.write(f"xform_out={output_trans_path}\n")
    regis_file.write(f"optim=rsg\n")
    regis_file.write(f"metric={metric}\n")
    regis_file.write(f"rsg_grad_tol={grad_tol}\n")
    regis_file.write(f"max_its={max_its}\n")
    regis_file.write(f"res={res_list[0]} {res_list[1]} {res_list[2]}\n")
    regis_file.close()
    os.system(f"plastimatch register {cmd_file_path}")
    # TODO:
    # affine registeration not work and produce trans with 1,0,0,0,1,0,0,0,1,0,0,0
    
    # affine segment transformation
    if(mov_seg_path is not None and output_seg_path is not None):
        assert(os.path.exists(mov_seg_path))
        mov_seg = read_mha_array3D(mov_seg_path).astype(np.float32)
        write_mha_array3D(mov_seg, mov_seg_path)
        make_parent_dir(output_seg_path)
        os.system(f"plastimatch warp --input {mov_seg_path} --output-img {output_seg_path} --xf {output_trans_path} --interpolation nn")

    # transfer input images
    assert len(transfer_input_image_paths)==len(transfer_output_image_paths)
    if(len(transfer_interpolation_methods)==0):
        transfer_interpolation_methods = ["nn"]*len(transfer_input_image_paths)
    for transfer_input_image_path, transfer_output_image_path, transfer_interpolation_method in \
        zip(transfer_input_image_paths, transfer_output_image_paths, transfer_interpolation_methods):
        plasimatch_apply_trans(
            transfer_input_image_path,
            output_trans_path,
            transfer_output_image_path,
            transfer_interpolation_method
        )

    if(remove_temp_dir):
        os.system(f"rm -r {temp_dir}")

def plastimatch_regis_demon(fix_image_path, 
                            mov_image_path,
                            output_image_path,
                            output_trans_path=None,
                            fix_mask_path=None,
                            mov_mask_path=None,
                            mov_seg_path=None,
                            output_seg_path=None,
                            default_value=0,
                            max_its=300,
                            transfer_input_image_paths=[],
                            transfer_output_image_paths=[],
                            transfer_interpolation_methods=[],
                            temp_dir="./temp_dir",
                            remove_temp_dir=True
                            ):
    """
    Perform demon registeration between target image(mov image) to template image(fix image).
    notice: 1.mask with value 1 is the region that consider during the plastimatch registering.
    notice: 2.output transform is of txt format.

    Args:
        fix_image_path (str): path of template image.
        mov_image_path (str): path of target image.
        output_image_path (str): path of output image from registeration.
        output_trans_path (str): path of output trans from registeration, txt format.
        fix_mask_path (str): path of template mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_mask_path (str): path of target mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_seg_path (str): path of target segment, used for segment transfer.
        output_seg_path (str): path of output template segment, transformed from mov_seg by output transformation.
        max_its: max iteration number.
        default_value (float): value to be pad in the region that newly produced by the registeration.
        transfer_input_image_paths (List): paths of other images to be transfered, must be the same length of transfer_output_image_paths.
        transfer_output_image_paths (List): paths of other images that output by warping transformation, must be the same length of transfer_input_image_paths.
        transfer_interpolation_methods (List): methods of interpolation to warp other images.
        temp_dir (str): temp directory to save plastimatch cmd file.
        remove_temp_dir (bool): whether to remove temp_dir after registeration.
    """
    # paths
    os.makedirs(temp_dir,exist_ok=True)
    cmd_file_path = f"{temp_dir}/cmd_file.txt"
    if(output_trans_path is None):
        output_trans_path = f"{temp_dir}/output_trans.nii.gz"
    make_parent_dir(output_image_path)
    make_parent_dir(output_trans_path)
    # demon registration    
    regis_file=open(cmd_file_path,mode="w")
    # global
    regis_file.write("[GLOBAL]\n")
    regis_file.write(f"fixed={fix_image_path}\n")
    regis_file.write(f"moving={mov_image_path}\n")
    if(fix_mask_path is not None):
        regis_file.write(f"fixed_mask={fix_mask_path}\n")
    if(mov_mask_path is not None):
        regis_file.write(f"moving_mask={mov_mask_path}\n")
    regis_file.write(f"img_out={output_image_path}\n")
    regis_file.write(f"xform_out={output_trans_path}\n")
    regis_file.write("\n")
    # stage
    regis_file.write("[STAGE]\n")
    regis_file.write("xform=vf\n")
    regis_file.write("impl=itk\n")
    regis_file.write("optim=demons\n")
    regis_file.write("optim_subtype=diffeomorphic\n")
    regis_file.write("demons_gradient_type=symmetric\n")
    regis_file.write("demons_step_length=1\n")
    regis_file.write(f"max_its={max_its}\n")
    regis_file.write("res=1 1 1\n")
    regis_file.close()
    os.system(f"plastimatch register {cmd_file_path}")
    
    # demon segment transformation
    if(mov_seg_path is not None and output_seg_path is not None):
        assert(os.path.exists(mov_seg_path))
        mov_seg = read_mha_array3D(mov_seg_path).astype(np.float32)
        write_mha_array3D(mov_seg, mov_seg_path)
        make_parent_dir(output_seg_path)
        os.system(f"plastimatch warp --input {mov_seg_path} --output-img {output_seg_path} --xf {output_trans_path} --interpolation nn")

    # transfer input images
    assert len(transfer_input_image_paths)==len(transfer_output_image_paths)
    if(len(transfer_interpolation_methods)==0):
        transfer_interpolation_methods = ["nn"]*len(transfer_input_image_paths)
    for transfer_input_image_path, transfer_output_image_path, transfer_interpolation_method in \
        zip(transfer_input_image_paths, transfer_output_image_paths, transfer_interpolation_methods):
        plasimatch_apply_trans(
            transfer_input_image_path,
            output_trans_path,
            transfer_output_image_path,
            transfer_interpolation_method
        )

    if(remove_temp_dir):
        os.system(f"rm -r {temp_dir}")

def plastimatch_regis_bspline(fix_image_path, 
                            mov_image_path,
                            output_image_path,
                            output_trans_path=None,
                            fix_mask_path=None,
                            mov_mask_path=None,
                            mov_seg_path=None,
                            output_seg_path=None,
                            default_value=0,
                            impl="itk",
                            threading=None,
                            metric=None,
                            max_its=300,
                            grad_tol=1e-6,
                            regularization_lambda=None,
                            res_list=[1,1,1],
                            grid_spac=[15,15,15],
                            transfer_input_image_paths=[],
                            transfer_output_image_paths=[],
                            transfer_interpolation_methods=[],
                            temp_dir="./temp_dir",
                            remove_temp_dir=True
                            ):
    """
    Perform demon registeration between target image(mov image) to template image(fix image).
    notice: 1.mask with value 1 is the region that consider during the plastimatch registering.
    notice: 2.output transform is of txt format.

    Args:
        fix_image_path (str): path of template image.
        mov_image_path (str): path of target image.
        output_image_path (str): path of output image from registeration.
        output_trans_path (str): path of output trans from registeration, txt format.
        fix_mask_path (str): path of template mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_mask_path (str): path of target mask, mask with value 1 is the region that consider during the plastimatch registering.
        mov_seg_path (str): path of target segment, used for segment transfer.
        output_seg_path (str): path of output template segment, transformed from mov_seg by output transformation.
        max_its: max iteration number.
        default_value (float): value to be pad in the region that newly produced by the registeration.
        transfer_input_image_paths (List): paths of other images to be transfered, must be the same length of transfer_output_image_paths.
        transfer_output_image_paths (List): paths of other images that output by warping transformation, must be the same length of transfer_input_image_paths.
        transfer_interpolation_methods (List): methods of interpolation to warp other images.
        temp_dir (str): temp directory to save plastimatch cmd file.
        remove_temp_dir (bool): whether to remove temp_dir after registeration.
    """
    # paths
    os.makedirs(temp_dir,exist_ok=True)
    cmd_file_path = f"{temp_dir}/cmd_file.txt"
    if(output_trans_path is None):
        output_trans_path = f"{temp_dir}/output_trans.txt"
    make_parent_dir(output_image_path)
    make_parent_dir(output_trans_path)
    # demon registration    
    regis_file=open(cmd_file_path,mode="w")
    
    # global
    regis_file.write("[GLOBAL]\n")
    regis_file.write(f"fixed={fix_image_path}\n")
    regis_file.write(f"moving={mov_image_path}\n")
    if(fix_mask_path is not None):
        regis_file.write(f"fixed_mask={fix_mask_path}\n")
    if(mov_mask_path is not None):
        regis_file.write(f"moving_mask={mov_mask_path}\n")
    regis_file.write(f"img_out={output_image_path}\n")
    regis_file.write(f"xform_out={output_trans_path}\n")
    regis_file.write(f"img_out={output_image_path}\n")
    regis_file.write("\n")
    # stage
    regis_file.write("[STAGE]\n")
    regis_file.write(f"impl={impl}\n")
    regis_file.write("xform=bspline\n")
    regis_file.write(f"xform_out={output_trans_path}\n")
    regis_file.write(f"optim=lbfgsb\n")
    if(metric is not None):
        regis_file.write(f"metric={metric}\n")
    if(regularization_lambda is not None):
        regis_file.write(f"regularization_lambda={regularization_lambda}\n")
    if(threading is not None):
        regis_file.write(f"threading={threading}\n")
    regis_file.write(f"pgtol={grad_tol}\n")
    regis_file.write(f"max_its={max_its}\n")
    regis_file.write(f"res={res_list[0]} {res_list[1]} {res_list[2]}\n")
    regis_file.write(f"grid_spac={grid_spac[0]} {grid_spac[1]} {grid_spac[2]}\n")
    regis_file.close()
    os.system(f"plastimatch register {cmd_file_path}")
    
    # demon segment transformation
    if(mov_seg_path is not None and output_seg_path is not None):
        assert(os.path.exists(mov_seg_path))
        mov_seg = read_mha_array3D(mov_seg_path).astype(np.float32)
        write_mha_array3D(mov_seg, mov_seg_path)
        make_parent_dir(output_seg_path)
        os.system(f"plastimatch warp --input {mov_seg_path} --output-img {output_seg_path} --xf {output_trans_path} --interpolation nn")

    # transfer input images
    assert len(transfer_input_image_paths)==len(transfer_output_image_paths)
    if(len(transfer_interpolation_methods)==0):
        transfer_interpolation_methods = ["nn"]*len(transfer_input_image_paths)
    for transfer_input_image_path, transfer_output_image_path, transfer_interpolation_method in \
        zip(transfer_input_image_paths, transfer_output_image_paths, transfer_interpolation_methods):
        plasimatch_apply_trans(
            transfer_input_image_path,
            output_trans_path,
            transfer_output_image_path,
            transfer_interpolation_method
        )

    if(remove_temp_dir):
        os.system(f"rm -r {temp_dir}")

def plastimatch_label_transfer(template_image_path,
                               template_seg_path,
                               target_image_path,
                               output_target_seg_path,
                               template_mask_path=None,
                               target_mask_path=None,
                               temp_dir="./transfer_temp_dir",
                               remove_temp_dir=True,
                               remove_rigid_result=True,
                               remove_affine_result=True,
                               remove_deform_result=True
                              ):
    '''
    Transfer label from mov_image to fix_image
    notice: 1.if target_mask_path is not given, will generate target_mask using detect_top and detect_bottom automatically.
    notice: 2.mask region with value 1 is the region that consider during the plastimatch registering.
    
    Args:
        template_image_path (str): path of template image
        template_seg_path (str): path of template segment
        target_image_path (str): path of image that needs segment
        output_target_seg_path (str): path of seg that transform output
        template_mask_path (str): path of mask for template image
        target_mask_path (str): path of tartget for target image
        temp_dir (str): dir to save intermediate file
        remove_temp_dir (bool): whether to remove temp dir

    Returns:
        None
    '''
    # path
    os.makedirs(temp_dir, exist_ok=True)
    # mask
    if(target_mask_path is None):
        target_mask_path = f"{temp_dir}/target_mask.nii.gz"
        target_image = read_mha_array3D(target_image_path)
        bottom_h = detect_bottom_array3D(target_image)
        top_h = detect_top_array3D(target_image)
        target_mask = np.zeros_like(target_image)
        target_mask[bottom_h:top_h,:,:]=1
        write_mha_array3D(target_mask, target_mask_path)
            
    # rigid
    rigid_template_image_path = f"{temp_dir}/rigid_template_image.nii.gz"
    rigid_template_seg_path = f"{temp_dir}/rigid_template_seg.nii.gz"
    rigid_trans_path = f"{temp_dir}/rigid_trans.txt"
    plastimatch_regis_rigid(
        fix_image_path = target_image_path,
        mov_image_path = template_image_path,
        mov_seg_path = template_seg_path,
        output_image_path = rigid_template_image_path,
        output_trans_path = rigid_trans_path,
        output_seg_path = rigid_template_seg_path,
        fix_mask_path = target_mask_path,
        temp_dir = f"{temp_dir}/temp_rigid",
        remove_temp_dir = remove_rigid_result
    )
    
    # affine
    affine_template_image_path = f"{temp_dir}/affine_template_image.nii.gz"
    affine_template_seg_path = f"{temp_dir}/affine_template_seg.nii.gz"
    affine_trans_path = f"{temp_dir}/affine_trans.txt"
    plastimatch_regis_affine(
        fix_image_path = target_image_path,
        mov_image_path = rigid_template_image_path,
        mov_seg_path = rigid_template_seg_path,
        output_image_path = affine_template_image_path,
        output_trans_path = affine_trans_path,
        output_seg_path = affine_template_seg_path,
        fix_mask_path = target_mask_path,
        temp_dir = f"{temp_dir}/temp_affine",
        remove_temp_dir = remove_affine_result
    )
    
    # deform
    deform_template_image_path = f"{temp_dir}/deform_template_image.nii.gz"
    deform_trans_path = f"{temp_dir}/deform_trans.nii.gz"
    plastimatch_regis_demon(
        fix_image_path = target_image_path,
        mov_image_path = affine_template_image_path,
        mov_seg_path = affine_template_seg_path,
        output_image_path = deform_template_image_path,
        output_trans_path = deform_trans_path,
        output_seg_path = output_target_seg_path,
        fix_mask_path = target_mask_path,
        temp_dir = f"{temp_dir}/temp_deform",
        remove_temp_dir = remove_deform_result
    )
    
    if(remove_temp_dir):
        os.system(f"rm -rf {os.path.abspath(temp_dir)}")

def plastimatch_warp_func(input_path,
                          output_path,
                          grid_path,
                          is_binary=False,
                          make_dirs=True
                          ):
    cmd = f"plastimatch warp --input {input_path} --output-img {output_path} --xf {grid_path}"
    if(is_binary):
        cmd += " --interpolation nn"
    if(make_dirs):
        make_parent_dir(output_path)
    ret = os.system(cmd)
    return ret

# sitk registration
def sitk_regis_rigid_get_transform(fix_image_path, mov_image_path, output_image_path=None, max_its=300):
    fix_cube = read_mha_array3D(fix_image_path)
    mov_cube = read_mha_array3D(mov_image_path)
    fix_image = itk.GetImageFromArray(fix_cube)
    mov_image = itk.GetImageFromArray(mov_cube)
    init_transform = itk.CenteredTransformInitializer(fix_image, mov_image, itk.Euler3DTransform(), itk.CenteredTransformInitializerFilter.GEOMETRY)
    regis=itk.ImageRegistrationMethod()
    regis.SetMetricAsMeanSquares()
    regis.SetInterpolator(itk.sitkNearestNeighbor)
    regis.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,minStep=1e-2,numberOfIterations=max_its)
    regis.SetOptimizerScalesFromPhysicalShift()
    regis.SetInitialTransform(init_transform)
    transform=regis.Execute(fixed=fix_image,moving=mov_image)
    trans_image=itk.Resample(mov_image,fix_image,transform,itk.sitkNearestNeighbor,0,mov_image.GetPixelID())
    if(output_image_path is not None):
        itk.WriteImage(trans_image, output_image_path)
    return transform

def ants_regis_syn(fix_image_path, 
                    mov_image_path,
                    output_image_path,
                    fix_mask_path=None,
                    mov_mask_path=None,
                    output_seg_path=None,
                    output_trans_path=None,
                    default_value=0,
                    max_its=(40, 20, 0),
                    transfer_input_image_paths=[],
                    transfer_output_image_paths=[],
                    transfer_interpolation_methods=[],
                    type_of_transform="SyNRA",
                    temp_dir="./",
                    clean_mat_file=True
                    ):
    """
        notes: type_of_transform possible value: SyNOnly, SyN, SyNRA
    """
    import ants
    ants_interpolation_method_dict = {
        "linear" : "linear",
        "nn" : "nearestNeighbor"
    }
    fix_image = ants.image_read(fix_image_path)
    mov_image = ants.image_read(mov_image_path)
    if(fix_mask_path is None):
        ret_trans = ants.registration(fixed=fix_image,
                                      moving=mov_image,
                                      reg_iterations=max_its,
                                      type_of_transform=type_of_transform)
    else:
        fix_mask = ants.image_read(fix_mask_path)
        ret_trans = ants.registration(fixed=fix_image,
                                      moving=mov_image,
                                      mask=fix_mask,
                                      reg_iterations=max_its,
                                      type_of_transform=type_of_transform)
    fwd_trans = ret_trans["fwdtransforms"]
    print(f"test ret_trans:{ret_trans}")
    # notes
    # SyNRA return fwd_trans is a list with two element, is geenral deform-field and an affine param
    if(output_trans_path is not None):
        shutil.copy(fwd_trans[0], output_trans_path)
    # print(f"test ants syn fwd_trans.shape:{len(fwd_trans)} {fwd_trans[0].shape}")
    # image
    output_image = ants.apply_transforms(fixed=fix_image, moving=mov_image, transformlist=fwd_trans)
    ants.image_write(output_image, output_image_path)
    for transfer_idx in range(0, len(transfer_input_image_paths)):
        transfer_input_image_path = transfer_input_image_paths[transfer_idx]
        transfer_output_image_path = transfer_output_image_paths[transfer_idx]
        transfer_interpolation_method = transfer_interpolation_methods[transfer_idx]
        transfer_interpolation_method = ants_interpolation_method_dict[transfer_interpolation_method]
        transfer_input_image = ants.image_read(transfer_input_image_path)
        transfer_output_image = ants.apply_transforms(fixed=fix_image,
                                                      moving=transfer_input_image,
                                                      transformlist=fwd_trans,
                                                      interpolator=transfer_interpolation_method)
        ants.image_write(transfer_output_image, transfer_output_image_path)

    # clean mat file
    # clean mat file
    if(clean_mat_file):
        for trans_name in ["fwdtransforms", "invtransforms"]:
            trans_file_path_list = ret_trans[trans_name]
            for trans_file_path in trans_file_path_list:
                if(os.path.exists(trans_file_path)):
                    os.remove(trans_file_path)

def ants_regis_rigid(fix_image_path, 
                    mov_image_path,
                    output_image_path,
                    fix_mask_path=None,
                    mov_mask_path=None,
                    output_seg_path=None,
                    output_trans_path=None,
                    default_value=0,
                    max_its=(40, 20, 0),
                    transfer_input_image_paths=[],
                    transfer_output_image_paths=[],
                    transfer_interpolation_methods=[],
                    type_of_transform="Rigid",
                    temp_dir="./",
                    clean_mat_file=True
                    ):
    """
        notes: 
        type_of_transform can be one of:
        - "Translation": Translation transformation.
        - "Rigid": Rigid transformation: Only rotation and translation.
        - "Similarity": Similarity transformation: scaling, rotation and translation.
        - "QuickRigid": Rigid transformation: Only rotation and translation.
                        May be useful for quick visualization fixes.'
        - "DenseRigid": Rigid transformation: Only rotation and translation.
                        Employs dense sampling during metric estimation.'
        - "BOLDRigid": Rigid transformation: Parameters typical for BOLD to
                        BOLD intrasubject registration'.'
        - "Affine": Affine transformation: Rigid + scaling.
        - "AffineFast": Fast version of Affine.
        - "BOLDAffine": Affine transformation: Parameters typical for BOLD to
                        BOLD intrasubject registration'.'
        - "TRSAA": translation, rigid, similarity, affine (twice). please set
                    regIterations if using this option. this would be used in
                    cases where you want a really high quality affine mapping
                    (perhaps with mask).
        - "Elastic": Elastic deformation: Affine + deformable.
        - "ElasticSyN": Symmetric normalization: Affine + deformable
                        transformation, with mutual information as optimization
                        metric and elastic regularization.
        - "SyN": Symmetric normalization: Affine + deformable transformation,
                    with mutual information as optimization metric.
        - "SyNRA": Symmetric normalization: Rigid + Affine + deformable
                    transformation, with mutual information as optimization metric.
        - "SyNOnly": Symmetric normalization: no initial transformation,
                    with mutual information as optimization metric. Assumes
                    images are aligned by an inital transformation. Can be
                    useful if you want to run an unmasked affine followed by
                    masked deformable registration.
        - "SyNCC": SyN, but with cross-correlation as the metric.
        - "SyNabp": SyN optimized for abpBrainExtraction.
        - "SyNBold": SyN, but optimized for registrations between BOLD and T1 images.
        - "SyNBoldAff": SyN, but optimized for registrations between BOLD
                        and T1 images, with additional affine step.
        - "SyNAggro": SyN, but with more aggressive registration
                        (fine-scale matching and more deformation).
                        Takes more time than SyN.
        - "TV[n]": time-varying diffeomorphism with where 'n' indicates number of
            time points in velocity field discretization.  The initial transform
            should be computed, if needed, in a separate call to ants.registration.
        - "TVMSQ": time-varying diffeomorphism with mean square metric
        - "TVMSQC": time-varying diffeomorphism with mean square metric for very large deformation
        - "antsRegistrationSyN[x]": recreation of the antsRegistrationSyN.sh script in ANTs
                                    where 'x' is one of the transforms available (e.g., 't', 'b', 's')
        - "antsRegistrationSyNQuick[x]": recreation of the antsRegistrationSyNQuick.sh script in ANTs
                                    where 'x' is one of the transforms available (e.g., 't', 'b', 's')
        - "antsRegistrationSyNRepro[x]": reproducible registration.  x options as above.
        - "antsRegistrationSyNQuickRepro[x]": quick reproducible registration.  x options as above.
    """
    import ants
    ants_interpolation_method_dict = {
        "linear" : "linear",
        "nn" : "nearestNeighbor"
    }
    fix_image = ants.image_read(fix_image_path)
    mov_image = ants.image_read(mov_image_path)
    if((fix_mask_path is not None) or (mov_mask_path is not None)):
        if(fix_mask_path is not None):
            mask = ants.image_read(fix_mask_path)
        elif(mov_mask_path is not None):
            mask = ants.image_read(mov_mask_path)
        ret_trans = ants.registration(fixed=fix_image,
                                      moving=mov_image,
                                      mask=mask,
                                      reg_iterations=max_its,
                                      type_of_transform=type_of_transform)
    else:
        ret_trans = ants.registration(fixed=fix_image,
                                      moving=mov_image,
                                      reg_iterations=max_its,
                                      type_of_transform=type_of_transform)

    fwd_trans = ret_trans["fwdtransforms"]
    # notes
    # SyNRA return fwd_trans is a list with two element, is geenral deform-field and an affine param
    if(output_trans_path is not None):
        shutil.copy(fwd_trans[0], output_trans_path)
    # print(f"test ants syn fwd_trans.shape:{len(fwd_trans)} {fwd_trans[0].shape}")
    # image
    output_image = ants.apply_transforms(fixed=fix_image, moving=mov_image, transformlist=fwd_trans)
    ants.image_write(output_image, output_image_path)
    for transfer_idx in range(0, len(transfer_input_image_paths)):
        transfer_input_image_path = transfer_input_image_paths[transfer_idx]
        transfer_output_image_path = transfer_output_image_paths[transfer_idx]
        transfer_interpolation_method = transfer_interpolation_methods[transfer_idx]
        transfer_interpolation_method = ants_interpolation_method_dict[transfer_interpolation_method]
        transfer_input_image = ants.image_read(transfer_input_image_path)
        transfer_output_image = ants.apply_transforms(fixed=fix_image,
                                                      moving=transfer_input_image,
                                                      transformlist=fwd_trans,
                                                      interpolator=transfer_interpolation_method)
        ants.image_write(transfer_output_image, transfer_output_image_path)
    
    # clean mat file
    if(clean_mat_file):
        for trans_name in ["fwdtransforms", "invtransforms"]:
            trans_file_path_list = ret_trans[trans_name]
            for trans_file_path in trans_file_path_list:
                if(os.path.exists(trans_file_path)):
                    os.remove(trans_file_path)
                    
def ants_regis_affine(fix_image_path, 
                    mov_image_path,
                    output_image_path,
                    fix_mask_path=None,
                    mov_mask_path=None,
                    output_seg_path=None,
                    output_trans_path=None,
                    default_value=0,
                    max_its=(40, 20, 0),
                    transfer_input_image_paths=[],
                    transfer_output_image_paths=[],
                    transfer_interpolation_methods=[],
                    type_of_transform="Affine",
                    temp_dir="./",
                    clean_mat_file=True
                    ):
    """
        notes: 
        type_of_transform can be one of:
        - "Translation": Translation transformation.
        - "Rigid": Rigid transformation: Only rotation and translation.
        - "Similarity": Similarity transformation: scaling, rotation and translation.
        - "QuickRigid": Rigid transformation: Only rotation and translation.
                        May be useful for quick visualization fixes.'
        - "DenseRigid": Rigid transformation: Only rotation and translation.
                        Employs dense sampling during metric estimation.'
        - "BOLDRigid": Rigid transformation: Parameters typical for BOLD to
                        BOLD intrasubject registration'.'
        - "Affine": Affine transformation: Rigid + scaling.
        - "AffineFast": Fast version of Affine.
        - "BOLDAffine": Affine transformation: Parameters typical for BOLD to
                        BOLD intrasubject registration'.'
        - "TRSAA": translation, rigid, similarity, affine (twice). please set
                    regIterations if using this option. this would be used in
                    cases where you want a really high quality affine mapping
                    (perhaps with mask).
        - "Elastic": Elastic deformation: Affine + deformable.
        - "ElasticSyN": Symmetric normalization: Affine + deformable
                        transformation, with mutual information as optimization
                        metric and elastic regularization.
        - "SyN": Symmetric normalization: Affine + deformable transformation,
                    with mutual information as optimization metric.
        - "SyNRA": Symmetric normalization: Rigid + Affine + deformable
                    transformation, with mutual information as optimization metric.
        - "SyNOnly": Symmetric normalization: no initial transformation,
                    with mutual information as optimization metric. Assumes
                    images are aligned by an inital transformation. Can be
                    useful if you want to run an unmasked affine followed by
                    masked deformable registration.
        - "SyNCC": SyN, but with cross-correlation as the metric.
        - "SyNabp": SyN optimized for abpBrainExtraction.
        - "SyNBold": SyN, but optimized for registrations between BOLD and T1 images.
        - "SyNBoldAff": SyN, but optimized for registrations between BOLD
                        and T1 images, with additional affine step.
        - "SyNAggro": SyN, but with more aggressive registration
                        (fine-scale matching and more deformation).
                        Takes more time than SyN.
        - "TV[n]": time-varying diffeomorphism with where 'n' indicates number of
            time points in velocity field discretization.  The initial transform
            should be computed, if needed, in a separate call to ants.registration.
        - "TVMSQ": time-varying diffeomorphism with mean square metric
        - "TVMSQC": time-varying diffeomorphism with mean square metric for very large deformation
        - "antsRegistrationSyN[x]": recreation of the antsRegistrationSyN.sh script in ANTs
                                    where 'x' is one of the transforms available (e.g., 't', 'b', 's')
        - "antsRegistrationSyNQuick[x]": recreation of the antsRegistrationSyNQuick.sh script in ANTs
                                    where 'x' is one of the transforms available (e.g., 't', 'b', 's')
        - "antsRegistrationSyNRepro[x]": reproducible registration.  x options as above.
        - "antsRegistrationSyNQuickRepro[x]": quick reproducible registration.  x options as above.
    """
    import ants
    ants_interpolation_method_dict = {
        "linear" : "linear",
        "nn" : "nearestNeighbor"
    }
    fix_image = ants.image_read(fix_image_path)
    mov_image = ants.image_read(mov_image_path)
    if((fix_mask_path is not None) or (mov_mask_path is not None)):
        if(fix_mask_path is not None):
            mask = ants.image_read(fix_mask_path)
        elif(mov_mask_path is not None):
            mask = ants.image_read(mov_mask_path)
        ret_trans = ants.registration(fixed=fix_image,
                                      moving=mov_image,
                                      mask=mask,
                                      reg_iterations=max_its,
                                      type_of_transform=type_of_transform)
    else:
        ret_trans = ants.registration(fixed=fix_image,
                                      moving=mov_image,
                                      reg_iterations=max_its,
                                      type_of_transform=type_of_transform)

    fwd_trans = ret_trans["fwdtransforms"]
    # notes
    # SyNRA return fwd_trans is a list with two element, is geenral deform-field and an affine param
    if(output_trans_path is not None):
        shutil.copy(fwd_trans[0], output_trans_path)
    # print(f"test ants syn fwd_trans.shape:{len(fwd_trans)} {fwd_trans[0].shape}")
    # image
    output_image = ants.apply_transforms(fixed=fix_image, moving=mov_image, transformlist=fwd_trans)
    ants.image_write(output_image, output_image_path)
    for transfer_idx in range(0, len(transfer_input_image_paths)):
        transfer_input_image_path = transfer_input_image_paths[transfer_idx]
        transfer_output_image_path = transfer_output_image_paths[transfer_idx]
        transfer_interpolation_method = transfer_interpolation_methods[transfer_idx]
        transfer_interpolation_method = ants_interpolation_method_dict[transfer_interpolation_method]
        transfer_input_image = ants.image_read(transfer_input_image_path)
        transfer_output_image = ants.apply_transforms(fixed=fix_image,
                                                      moving=transfer_input_image,
                                                      transformlist=fwd_trans,
                                                      interpolator=transfer_interpolation_method)
        ants.image_write(transfer_output_image, transfer_output_image_path)
    
    # clean mat file
    if(clean_mat_file):
        for trans_name in ["fwdtransforms", "invtransforms"]:
            trans_file_path_list = ret_trans[trans_name]
            for trans_file_path in trans_file_path_list:
                if(os.path.exists(trans_file_path)):
                    os.remove(trans_file_path)

def ants_invert_trans(
                    input_trans_path,
                    inverse_trans_path,
                    debug=False
                    ):
    import ants
    input_trans = ants.read_transform(filename=input_trans_path)
    invert_trans = input_trans.invert()
    make_parent_dir(inverse_trans_path)
    ants.write_transform(transform=invert_trans, filename=inverse_trans_path)
    if(debug):
        read_invert_trans = ants.read_transform(inverse_trans_path)
        print(f"input_trans:\n{input_trans.parameters}")
        print(f"invert_trans:\n{read_invert_trans.parameters}")
        

def ants_apply_trans(
                    input_image_path,
                    input_trans_path,
                    output_image_path,
                    interpolation_type="nn"
                    ):
    """
    Perform transformation over the input image with the given transformation file path.

    Args:
        input_image_path (str): path of input image.
        input_trans_path (str): path of input trans.
        output_image_path (str): path of output image.
        interpolation_type (str, optional): interpolation type. Defaults to "nn".
    """
    import ants
    ants_interpolation_method_dict = {
        "linear" : "linear",
        "nn" : "nearestNeighbor"
    }
    input_image = ants.image_read(filename=input_image_path)
    fix_image = copy.copy(input_image)
    interpolation_method = ants_interpolation_method_dict[interpolation_type]
    output_image = ants.apply_transforms(
        fixed=fix_image,
        moving=input_image,
        transformlist=[input_trans_path],
        interpolator=interpolation_method
    )
    make_parent_dir(output_image_path)
    ants.image_write(output_image, output_image_path)
    return True

def ants_apply_trans_list(
    input_image_path_list,
    input_trans_path_list,
    output_image_path_list,
    interpolation_type_list=None):
    """
    Perform transformation over the input image with the given transformation file path.

    Args:
        input_image_path_list (List[str]): list of path of input image.
        input_trans_path_list (List[str]): list of path of input trans.
        output_image_path_list (List[str]): path of output image.
        interpolation_type_list (List[str], optional): interpolation type. Defaults to "nn".
    """
    if(interpolation_type_list is None):
        interpolation_type_list = ["linear"] * len(input_image_path_list)
    assert(len(interpolation_type_list)==len(input_image_path_list))
    warp_num = len(input_image_path_list)
    for warp_idx in range(0, warp_num):
        ants_apply_trans(
            input_image_path=input_image_path_list[warp_idx],
            input_trans_path=input_trans_path_list[warp_idx],
            output_image_path=output_image_path_list[warp_idx],
            interpolation_type=interpolation_type_list[warp_idx]
        )
    return True
    

def center_align_by_center(
    center_h,
    center_w,
    center_l,
    mov_seg_path,
    output_seg_path=None,
    mov_image_path=None,
    output_image_path=None,
    ret_image=True
):
    fix_seg_centroid = np.array([center_h, center_w, center_l])
    mov_seg = read_mha_array3D(mov_seg_path)
    mov_seg_centroid= np.array(get_centroid(mov_seg))
    trans_h, trans_w, trans_l = fix_seg_centroid - mov_seg_centroid
    align_dict = {
        "trans_x" : trans_h,
        "trans_y" : trans_w,
        "trans_z" : trans_l
    }
    
    # output seg
    output_seg = spatial_transform_by_param_array3D(
        image = mov_seg,
        trans_x = trans_h,
        trans_y = trans_w,
        trans_z = trans_l,
        mode = "nearest"
    )
    if(output_seg_path is not None):
        write_mha_array3D(output_seg, output_seg_path)
        
    if(mov_image_path is not None):
        mov_image = read_mha_array3D(mov_image_path)
        output_image = spatial_transform_by_param_array3D(
            image = mov_image,
            trans_x = trans_h,
            trans_y = trans_w,
            trans_z = trans_l,
            mode = "bilinear"
        )
        if(output_image_path is not None):
            write_mha_array3D(output_image, output_image_path)
    return align_dict

def center_align_by_seg(
    fix_seg_path,
    mov_seg_path,
    output_seg_path=None,
    mov_image_path=None,
    output_image_path=None,
):
    fix_seg = read_mha_array3D(fix_seg_path)
    mov_seg = read_mha_array3D(mov_seg_path)
    fix_seg_centroid = np.array(get_centroid(fix_seg))
    mov_seg_centroid= np.array(get_centroid(mov_seg))
    trans_h, trans_w, trans_l = fix_seg_centroid - mov_seg_centroid
    
    if(output_seg_path is not None):
        output_seg = spatial_transform_by_param_array3D(
        image = mov_seg,
        trans_x = trans_h,
        trans_y = trans_w,
        trans_z = trans_l,
        mode = "nearest"
        )
        write_mha_array3D(output_seg, output_seg_path)
        
    if((mov_image_path is not None) and (output_image_path is not None)):
        mov_image = read_mha_array3D(mov_image_path)
        output_image = spatial_transform_by_param_array3D(
            image = mov_image,
            trans_x = trans_h,
            trans_y = trans_w,
            trans_z = trans_l,
            mode = "bilinear"
        )
        write_mha_array3D(output_image, output_image_path)

    align_dict = {
        "trans_x" : trans_h,
        "trans_y" : trans_w,
        "trans_z" : trans_l
    }
    return align_dict

def center_align_by_seg_get_align_dict(
    fix_seg_path,
    mov_seg_path
):
    fix_seg = read_mha_array3D(fix_seg_path)
    mov_seg = read_mha_array3D(mov_seg_path)
    fix_seg_centroid = np.array(get_centroid(fix_seg))
    mov_seg_centroid= np.array(get_centroid(mov_seg))
    trans_h, trans_w, trans_l = fix_seg_centroid - mov_seg_centroid
    align_dict = {
        "trans_x" : trans_h,
        "trans_y" : trans_w,
        "trans_z" : trans_l
    }
    return align_dict

def center_align_by_align_dict(
    align_dict,
    mov_seg_path,
    output_seg_path=None,
    mov_image_path=None,
    output_image_path=None
):
    mov_seg = read_mha_array3D(mov_seg_path)
    trans_h = align_dict["trans_x"]
    trans_w = align_dict["trans_y"]
    trans_l = align_dict["trans_z"]
    
    if(output_seg_path is not None):
        output_seg = spatial_transform_by_param_array3D(
        image = mov_seg,
        trans_x = trans_h,
        trans_y = trans_w,
        trans_z = trans_l,
        mode = "nearest"
        )
        write_mha_array3D(output_seg, output_seg_path)
        
    if((mov_image_path is not None) and (output_image_path is not None)):
        mov_image = read_mha_array3D(mov_image_path)
        output_image = spatial_transform_by_param_array3D(
            image = mov_image,
            trans_x = trans_h,
            trans_y = trans_w,
            trans_z = trans_l,
            mode = "bilinear"
        )
        write_mha_array3D(output_image, output_image_path)

    return align_dict 

def center_align_list_by_seg(
    fix_seg_path,
    mov_seg_path,
    output_seg_path=None,
    transfer_input_image_paths=[],
    transfer_output_image_paths=[],
    transfer_interpolation_methods=[]
):
    # interpolation method dict
    spatial_trans_method_dict = {
        "linear" : "bilinear",
        "nn" : "nearest",
        "bicubic" : "bicubic"
    }
    
    fix_seg = read_mha_array3D(fix_seg_path)
    mov_seg = read_mha_array3D(mov_seg_path)
    fix_seg_centroid = np.array(get_centroid(fix_seg))
    mov_seg_centroid= np.array(get_centroid(mov_seg))
    trans_h, trans_w, trans_l = fix_seg_centroid - mov_seg_centroid
    
    if(output_seg_path is not None):
        output_seg = spatial_transform_by_param_array3D(
        image = mov_seg,
        trans_x = trans_h,
        trans_y = trans_w,
        trans_z = trans_l,
        mode = "nearest"
        )
        write_mha_array3D(output_seg, output_seg_path)
    
    for trans_idx in range(0, len(transfer_input_image_paths)):
        input_image_path = transfer_input_image_paths[trans_idx]
        output_image_path = transfer_output_image_paths[trans_idx]
        spatial_trans_method = spatial_trans_method_dict[transfer_interpolation_methods[trans_idx]]
        input_image = read_mha_array3D(input_image_path)
        output_image = spatial_transform_by_param_array3D(
            image = input_image,
            trans_x = trans_h,
            trans_y = trans_w,
            trans_z = trans_l,
            mode = spatial_trans_method
        )
        write_mha_array3D(output_image, output_image_path)

    align_dict = {
        "trans_x" : trans_h,
        "trans_y" : trans_w,
        "trans_z" : trans_l
    }
    return align_dict
    
    
