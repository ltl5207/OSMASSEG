import functools
from .dependencies import *

# basic type conversion
def totensor(array_list):
    if(type(array_list)==torch.Tensor):
        ret_tensor = array_list
    elif(type(array_list)==torch.FloatTensor):
        ret_tensor = array_list
    elif(type(array_list)==np.ndarray):
        ret_tensor = torch.FloatTensor(array_list)
    elif(type(array_list)==list):
        tensor_list = []
        for array in array_list:
            tensor_list.append(torch.FloatTensor(array))
        ret_tensor = tensor_list
    else:
        raise NotImplementedError(f"ToTensor Func not implemented type:{type(ret_tensor)}")
    # print(f"test totensor array_list:{type(array_list)} ret_tensor:{type(ret_tensor)}")
    return ret_tensor

def tonumpy(tensor_list):
    return map(lambda x: x.cpu().numpy(),tensor_list)

def tocuda(tensor_list):
    return map(lambda x: x.cuda(),tensor_list)

def tocuda_dict(tensor_dict):
    for name in tensor_dict.keys():
        if(hasattr(tensor_dict[name],"cuda")):
            tensor_dict[name] = tensor_dict[name].cuda()
    return tensor_dict

def get_full_kernel3():
    return torch.ones(size=[1,1,3,3,3])

def get_register_inv_tensor(x):
    batch_size=x.shape[0]
    trans_x=torch.zeros_like(x)
    for batch_idx in range(0,batch_size):
        ori_cube=x[batch_idx,0,:,:,:].detach().cpu().numpy()
        ori_image=itk.GetImageFromArray(ori_cube)
        inv_cube=ori_cube[:,:,::-1]
        inv_image=itk.GetImageFromArray(inv_cube)
        init_transform = itk.CenteredTransformInitializer(ori_image, inv_image, itk.Euler3DTransform(), itk.CenteredTransformInitializerFilter.GEOMETRY)
        regis=itk.ImageRegistrationMethod()
        regis.SetMetricAsMeanSquares()
        regis.SetInterpolator(itk.sitkLinear)
        regis.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,minStep=1e-2,numberOfIterations=100)
        regis.SetOptimizerScalesFromPhysicalShift()
        regis.SetInitialTransform(init_transform)
        transform=regis.Execute(fixed=ori_image,moving=inv_image)
        transed_image=itk.Resample(inv_image,ori_image,transform,itk.sitkLinear,(-999.0+1000)/2300,inv_image.GetPixelID())
        trans_cube=itk.GetArrayFromImage(transed_image)
        trans_x[batch_idx,0,:,:,:]=torch.FloatTensor(trans_cube)
    trans_x.requires_grad=False
    return trans_x

def get_var(x):
    #conv x
    conv_x_weight=torch.zeros(size=(1,1,3,3,3)).to(x.device)
    conv_x_weight[0,0,0,1,1]=-1
    conv_x_weight[0,0,2,1,1]=1
    grad_x=F.conv3d(input=x,weight=conv_x_weight,bias=None,stride=1,padding=1)
    #conv y
    conv_y_weight=torch.zeros(size=(1,1,3,3,3)).to(x.device)
    conv_y_weight[0,0,1,0,1]=-1
    conv_y_weight[0,0,1,2,1]=1
    grad_y=F.conv3d(input=x,weight=conv_y_weight,bias=None,stride=1,padding=1)
    #conv z
    conv_z_weight=torch.zeros(size=(1,1,3,3,3)).to(x.device)
    conv_z_weight[0,0,1,1,0]=-1
    conv_z_weight[0,0,1,1,2]=1
    grad_z=F.conv3d(input=x,weight=conv_z_weight,bias=None,stride=1,padding=1)
    #conv norm
    grad_norm=(grad_x**2+grad_y**2+grad_z**2)**0.5
    x_var=torch.stack([grad_x,grad_y,grad_z,grad_norm],dim=1)
    return x_var

#bouding box
def get_mask_bbx3D_tensor(label_cube,margin=4):
    #get bounding box of the binary mask
    #input [h,w,l] tensor
    mask=torch.zeros_like(label_cube)
    if(torch.sum(label_cube)!=0):
        img_h,img_w,img_l=label_cube.shape
        hs,ws,ls=torch.where(label_cube>0)
        mask_hmin=torch.clip(torch.min(hs)-margin,0,img_h)
        mask_hmax=torch.clip(torch.max(hs)+margin,0,img_h)
        mask_wmin=torch.clip(torch.min(ws)-margin,0,img_w)
        mask_wmax=torch.clip(torch.max(ws)+margin,0,img_w)
        mask_lmin=torch.clip(torch.min(ls)-margin,0,img_l)
        mask_lmax=torch.clip(torch.max(ls)+margin,0,img_l)
        mask[mask_hmin:mask_hmax,mask_wmin:mask_wmax,mask_lmin:mask_lmax]=1
    mask=mask.astype(torch.float32)
    return mask

def get_mask_bbx4D_tensor(bin_mask,pad_h=5,pad_w=5,pad_l=5):
    #get bounding box of the binary mask
    #input [c,h,w,l] tensor
    mask=torch.zeros_like(bin_mask)
    if(torch.sum(bin_mask>0)!=0):
        _,h,w,l=bin_mask.shape
        _,hs,ws,ls=torch.where(bin_mask>0)
        min_h,max_h=torch.min(hs),torch.max(hs)
        min_w,max_w=torch.min(ws),torch.max(ws)
        min_l,max_l=torch.min(ls),torch.max(ls)
        min_h=torch.clip(min_h-pad_h,0,h)
        max_h=torch.clip(max_h+pad_h,0,h)
        min_w=torch.clip(min_w-pad_w,0,w)
        max_w=torch.clip(max_w+pad_w,0,w)
        min_l=torch.clip(min_l-pad_l,0,l)
        max_l=torch.clip(max_l+pad_l,0,l)
        mask[:,min_h:max_h,min_w:max_w,min_l:max_l]=1
    return mask

def get_mask_bbx4D_tensor_batch(bin_mask_batch, pad_h=5, pad_w=5, pad_l=5):
    batch_size = bin_mask_batch.shape[0]
    bbx_mask_list = []
    for batch_idx in range(0, batch_size):
        bbx_mask_list.append(get_mask_bbx4D_tensor(bin_mask_batch[batch_idx], pad_h=pad_h, pad_w=pad_w, pad_l=pad_l))
    return torch.stack(bbx_mask_list)

def get_bbx_boundry_tensor4D(bbx_mask):
    """
    get the boundry of the given bbx_mask

    Args:
        bbx_mask (tensor): shape [c,h,w,l], binary.
    
    Returns:
        min_h (tensor): min_h.
        max_h (tensor): max_h.
        min_w (tensor): min_w.
        max_w (tensor): max_w.
        min_l (tensor): min_l.
        max_l (tensor): max_l.
    """
    _, hs, ws, ls = torch.where(bbx_mask>0)
    min_h, max_h = torch.min(hs),torch.max(hs)
    min_w, max_w = torch.min(ws),torch.max(ws)
    min_l, max_l = torch.min(ls),torch.max(ls)
    return min_h, max_h, min_w, max_w, min_l, max_l

def get_bbx_range_tensor4D(bbx_mask):
    """
    get the boundry of the given bbx_mask

    Args:
        bbx_mask (tensor): shape [c,h,w,l], binary.
    
    Returns:
        range_h (tensor): range of bbx in h-dimension.
        range_w (tensor): range of bbx in w-dimension.
        range_l (tensor): range of bbx in l-dimension
    """
    min_h, max_h, min_w, max_w, min_l, max_l = get_bbx_boundry_tensor4D(bbx_mask=bbx_mask)
    range_h = max_h - min_h + 1
    range_w = max_w - min_w + 1
    range_l = max_l - min_l + 1
    return range_h, range_w, range_l

def cal_dice_tensor(gt_label, pd_label):
    cal_gt_label=torch.where(gt_label!=0,1,0)
    cal_pd_label=torch.where(pd_label!=0,1,0)
    intersec = cal_gt_label*cal_pd_label
    dice=2*torch.sum(intersec)/torch.clamp(torch.sum(cal_gt_label)+torch.sum(cal_pd_label), min=1e-5)
    return dice

def get_local_patch(x,mask,patch_h=64,patch_w=64,patch_l=64):
    #get local patch of the binary mask
    #input [c,h,w,l] tensor
    _,h,w,l=mask.shape
    if(torch.sum(mask)!=0):
        _,hs,ws,ls=torch.where(mask!=0)
        min_h,max_h=torch.min(hs),torch.max(hs)
        min_w,max_w=torch.min(ws),torch.max(ws)
        min_l,max_l=torch.min(ls),torch.max(ls)
        start_h=min_h
        start_w=min_w
        start_l=min_l
        if(start_h+patch_h>=h):
            start_h=max_h-patch_h
        if(start_w+patch_w>=w):
            start_w=max_w-patch_w
        if(start_l+patch_l>=l):
            start_l=max_l-patch_l
    else:
        start_h=h-patch_h//2
        start_w=w-patch_w//2
        start_l=l-patch_l//2
    patch_x=x[:,start_h:start_h+patch_h,start_w:start_w+patch_w,start_l:start_l+patch_l]
    patch_mask=mask[:,start_h:start_h+patch_h,start_w:start_w+patch_w,start_l:start_l+patch_l]
    return patch_x,patch_mask

def is_tensor(x):
    return type(x) == torch.FloatTensor or type(x) == torch.LongTensor or type(x) == torch.Tensor

def to_tensor_scalar(scalar):
    return torch.ones(size=[])*scalar

def to_tensor_list(number_list):
    tensor_list = [torch.FloatTensor([number]) for number in number_list]
    return tensor_list

def totensor_dict(tensor_dict):
    for key in tensor_dict.keys():
        tensor_dict[key] = torch.FloatTensor([tensor_dict[key]])
    return tensor_dict

def get_grid_tensor(shape):
    '''
    Parameters:
        shape: shape [3], (h,w,l) of image thats need grid
    Returns:
        grid: shape [dim,h,w,l], dim=3
    '''
    vectors = [torch.arange(0, s) for s in shape]
    grid = torch.meshgrid(vectors)
    grid = torch.stack(grid)
    grid = grid.float()
    return grid

@functools.lru_cache(maxsize=8)
def get_grid_tensor_by_param(h, w, l):
    return get_grid_tensor(shape=[h, w, l])    

def get_inv_rigid_matrix_batch(init_rigid_matrix_batch):
    batch_num = init_rigid_matrix_batch.shape[0]
    inv_rigid_matrix_list = []
    for batch_idx in range(0, batch_num):
        inv_rigid_matrix = get_inv_rigid_matrix(init_rigid_matrix=init_rigid_matrix_batch[batch_idx])
        inv_rigid_matrix_list.append(inv_rigid_matrix)
    inv_rigid_matrix_batch = torch.stack(inv_rigid_matrix_list)
    return inv_rigid_matrix_batch

def get_inv_rigid_matrix(init_rigid_matrix):
    # print(f"test type:{type(init_rigid_matrix)} shape:{init_rigid_matrix.shape} device:{init_rigid_matrix.device}")
    if(not isinstance(init_rigid_matrix, torch.Tensor)):
        rigid_matrix = torch.FloatTensor(init_rigid_matrix)
    else:
        rigid_matrix = init_rigid_matrix
    inv_rigid_matrix = torch.zeros_like(rigid_matrix).to(rigid_matrix.device)
    rotate_matrix = rigid_matrix[:3,:3]
    trans_vec = rigid_matrix[:3, 3]
    inv_rigid_matrix[:3, :3] = rotate_matrix.transpose(0,1)
    inv_rigid_matrix[:3, 3] = -(rotate_matrix.transpose(0,1))@trans_vec
    inv_rigid_matrix[3, 3] = 1
    return inv_rigid_matrix
    
def check_cvt_tensor_scalar(x):
    if(not is_tensor(x)):
        return to_tensor_scalar(x)
    else:
        return x

def get_rigid_matrix_from_param(trans_x,trans_y,trans_z,roll,pitch,yaw,center_x=63.5,center_y=63.5,center_z=63.5,debug=False):
    '''
    Parameters:
        trans_x: shape [1]
        trans_y: shape [1]
        trans_z: shape [1]
        roll: shape [1]
        pitch: shape [1]
        yaw: shape [1]
    Returns:
        rigid_matrix: shape [4,4], rigid transformation matrix in homogeneous form
    '''
    # input trans_x,trans_y,trans_z,roll,pitch,yaw
    # output rigid matrix of shape [4,4]
    # print(f"test type: trans_x:{type(trans_x)} trans_y:{type(trans_y)} trans_z:{type(trans_z)} roll:{type(roll)} pitch:{type(pitch)} yaw:{type(yaw)}")
    
    if(not is_tensor(trans_x)):
        trans_x = to_tensor_scalar(trans_x)

    if(not is_tensor(trans_y)):
        trans_y = to_tensor_scalar(trans_y)

    if(not is_tensor(trans_z)):
        trans_z = to_tensor_scalar(trans_z)

    if(not is_tensor(roll)):
        roll = to_tensor_scalar(roll)

    if(not is_tensor(pitch)):
        pitch = to_tensor_scalar(pitch)

    if(not is_tensor(yaw)):
        yaw = to_tensor_scalar(yaw)

    from torch import cos,sin
    device = trans_x.device
    torch.FloatTensor()
    # center
    Rc = torch.zeros(size=[4,4]).to(device)
    Rc[0,0] = 1
    Rc[1,1] = 1
    Rc[2,2] = 1
    Rc[0,3] = -center_x
    Rc[1,3] = -center_y
    Rc[2,3] = -center_z
    Rc[3,3] = 1
    
    # rotation
    # Rx
    Rx=torch.zeros(size=[4,4]).to(device)
    Rx[0,0]=1
    Rx[1,1]=cos(roll)
    Rx[1,2]=sin(roll)
    Rx[2,1]=-sin(roll)
    Rx[2,2]=cos(roll)
    Rx[3,3]=1
    # Ry
    Ry=torch.zeros(size=[4,4]).to(device)
    Ry[0,0]=cos(pitch)
    Ry[0,2]=-sin(pitch)
    Ry[1,1]=1
    Ry[2,0]=sin(pitch)
    Ry[2,2]=cos(pitch)
    Ry[3,3]=1
    # Rz
    Rz=torch.zeros(size=[4,4]).to(device)
    Rz[0,0]=cos(yaw)
    Rz[0,1]=sin(yaw)
    Rz[1,0]=-sin(yaw)
    Rz[1,1]=cos(yaw)
    Rz[2,2]=1
    Rz[3,3]=1
    # combine
    Rr = Rx@Ry@Rz
    
    # transformation
    Rt = torch.zeros(size=[4,4]).to(device)
    Rt[0,0] = 1
    Rt[1,1] = 1
    Rt[2,2] = 1
    Rt[0,3] = trans_x
    Rt[1,3] = trans_y
    Rt[2,3] = trans_z
    Rt[3,3] = 1

    # de-center
    Rdc = torch.zeros(size=[4,4]).to(device)
    Rdc[0,0] = 1
    Rdc[1,1] = 1
    Rdc[2,2] = 1
    Rdc[0,3] = center_x
    Rdc[1,3] = center_y
    Rdc[2,3] = center_z
    Rdc[3,3] = 1
        
    # total
    R = Rt@Rdc@Rr@Rc
    #print(f"test Rr:\n{Rr}\nR:\n{R}")
    
    if(debug):
        print(f"[Debug] get_rigid_matrix_from_param: ")
        print(f"trans_x:{trans_x} trans_y:{trans_y} trans_z:{trans_z} roll:{roll} pitch:{pitch} yaw:{yaw} center_x:{center_x} center_y:{center_y} center_z:{center_z}\n")
        print(f"result matrix:\n{R}\n")
    return R

def get_inv_rigid_matrix_from_param(trans_x,trans_y,trans_z,roll,pitch,yaw,center_x=63.5,center_y=63.5,center_z=63.5):
    '''
    Parameters:
        trans_x: shape [1]
        trans_y: shape [1]
        trans_z: shape [1]
        roll: shape [1]
        pitch: shape [1]
        yaw: shape [1]
    Returns:
        rigid_matrix: shape [4,4], rigid transformation matrix in homogeneous form
    '''
    # input trans_x,trans_y,trans_z,roll,pitch,yaw
    # output rigid matrix of shape [4,4]
    print(f"test get_rigid_matrix trans_x:{trans_x} trans_y:{trans_y} trans_z:{trans_z} roll:{roll} pitch:{pitch} yaw:{yaw} center_x:{center_x} center_y:{center_y} center_z:{center_z}")
    if(not is_tensor(trans_x)):
        trans_x,trans_y,trans_z,roll,pitch,yaw = to_tensor_list([trans_x,trans_y,trans_z,roll,pitch,yaw])
    from torch import cos,sin
    device = trans_x.device
    # center
    Rc = torch.zeros(size=[4,4]).to(device)
    Rc[0,0] = 1
    Rc[1,1] = 1
    Rc[2,2] = 1
    Rc[0,3] = -center_x
    Rc[1,3] = -center_y
    Rc[2,3] = -center_z
    Rc[3,3] = 1
    
    # rotation
    # Rx
    Rx=torch.zeros(size=[4,4]).to(device)
    Rx[0,0]=1
    Rx[1,1]=cos(roll)
    Rx[1,2]=sin(roll)
    Rx[2,1]=-sin(roll)
    Rx[2,2]=cos(roll)
    Rx[3,3]=1
    # Ry
    Ry=torch.zeros(size=[4,4]).to(device)
    Ry[0,0]=cos(pitch)
    Ry[0,2]=-sin(pitch)
    Ry[1,1]=1
    Ry[2,0]=sin(pitch)
    Ry[2,2]=cos(pitch)
    Ry[3,3]=1
    # Rz
    Rz=torch.zeros(size=[4,4]).to(device)
    Rz[0,0]=cos(yaw)
    Rz[0,1]=sin(yaw)
    Rz[1,0]=-sin(yaw)
    Rz[1,1]=cos(yaw)
    Rz[2,2]=1
    Rz[3,3]=1
    # combine
    Rr = Rx@Ry@Rz
    inv_Rr = Rz@Ry@Rx
    
    # transformation
    Rt = torch.zeros(size=[4,4]).to(device)
    Rt[0,0] = 1
    Rt[1,1] = 1
    Rt[2,2] = 1
    Rt[0,3] = trans_x
    Rt[1,3] = trans_y
    Rt[2,3] = trans_z
    Rt[3,3] = 1

    # de-center
    Rdc = torch.zeros(size=[4,4]).to(device)
    Rdc[0,0] = 1
    Rdc[1,1] = 1
    Rdc[2,2] = 1
    Rdc[0,3] = center_x
    Rdc[1,3] = center_y
    Rdc[2,3] = center_z
    Rdc[3,3] = 1
        
    # total
    R = Rt@Rdc@Rr@Rc
    inv_R = Rc@inv_Rr@Rdc@(-Rt)
    print(f"test R@inv_R:{R@inv_R}")
    return inv_R

def get_rigid_matrix_from_param_ZYX(trans_x,trans_y,trans_z,roll,pitch,yaw,center_x=63.5,center_y=63.5,center_z=63.5):
    '''
    Parameters:
        trans_x: shape [1]
        trans_y: shape [1]
        trans_z: shape [1]
        roll: shape [1]
        pitch: shape [1]
        yaw: shape [1]
    Returns:
        rigid_matrix: shape [4,4], rigid transformation matrix in homogeneous form
    '''
    # input trans_x,trans_y,trans_z,roll,pitch,yaw
    # output rigid matrix of shape [4,4]
    if(not is_tensor(trans_x)):
        trans_x,trans_y,trans_z,roll,pitch,yaw = to_tensor_list([trans_x,trans_y,trans_z,roll,pitch,yaw])
    from torch import cos,sin
    device = trans_x.device
    # center
    Rc = torch.zeros(size=[4,4]).to(device)
    Rc[0,0] = 1
    Rc[1,1] = 1
    Rc[2,2] = 1
    Rc[0,3] = -center_x
    Rc[1,3] = -center_y
    Rc[2,3] = -center_z
    Rc[3,3] = 1
    
    # rotation
    # Rx
    Rx=torch.zeros(size=[4,4]).to(device)
    Rx[0,0]=1
    Rx[1,1]=cos(roll)
    Rx[1,2]=sin(roll)
    Rx[2,1]=-sin(roll)
    Rx[2,2]=cos(roll)
    Rx[3,3]=1
    # Ry
    Ry=torch.zeros(size=[4,4]).to(device)
    Ry[0,0]=cos(pitch)
    Ry[0,2]=-sin(pitch)
    Ry[1,1]=1
    Ry[2,0]=sin(pitch)
    Ry[2,2]=cos(pitch)
    Ry[3,3]=1
    # Rz
    Rz=torch.zeros(size=[4,4]).to(device)
    Rz[0,0]=cos(yaw)
    Rz[0,1]=sin(yaw)
    Rz[1,0]=-sin(yaw)
    Rz[1,1]=cos(yaw)
    Rz[2,2]=1
    Rz[3,3]=1
    # combine
    Rr = Rz@Ry@Rx
    
    # transformation
    Rt = torch.zeros(size=[4,4]).to(device)
    Rt[0,0] = 1
    Rt[1,1] = 1
    Rt[2,2] = 1
    Rt[0,3] = trans_x
    Rt[1,3] = trans_y
    Rt[2,3] = trans_z
    Rt[3,3] = 1

    # de-center
    Rdc = torch.zeros(size=[4,4]).to(device)
    Rdc[0,0] = 1
    Rdc[1,1] = 1
    Rdc[2,2] = 1
    Rdc[0,3] = center_x
    Rdc[1,3] = center_y
    Rdc[2,3] = center_z
    Rdc[3,3] = 1
        
    # total
    R = Rt@Rdc@Rr@Rc
    return R

def get_rigid_matrix_from_param_batch(trans_x,trans_y,trans_z,roll,pitch,yaw, center_x=63.5, center_y=63.5, center_z=63.5):
    '''
    Parameters:
        trans_x: shape [n,1]
        trans_y: shape [n,1]
        trans_z: shape [n,1]
        roll: shape [n,1]
        pitch: shape [n,1]
        yaw: shape [n,1]
    Returns:
        rigid_matrix: shape [4,4], rigid transformation matrix in homogeneous form
    '''
    batch_size = trans_x.shape[0]
    rigid_matrix = torch.zeros(size=[batch_size,4,4]).to(trans_x.device)
    for batch_idx in range(0,batch_size):
        rigid_matrix[batch_idx,:,:]=get_rigid_matrix_from_param(trans_x[batch_idx], trans_y[batch_idx], trans_z[batch_idx],\
            roll[batch_idx], pitch[batch_idx], yaw[batch_idx], center_x=center_x, center_y=center_y, center_z=center_z)
    return rigid_matrix

def get_deform_from_rigid_matrix(rigid_matrix, shape):
    '''
    Parameters:
        rigid_matrix: shape [4,4], homo transform
        shape: shape [3], shape of image
    Returns:
        deform: shape [dim,h,w,l], output deformation field 
    '''
    h,w,l = shape
    dim = len(shape)
    device = rigid_matrix.device
    # create sampling grid of shape [h,w,l,dim]
    grid =  get_grid_tensor(shape=shape).to(device).permute(1,2,3,0)
    # create deform of shape [h,w,l,dim+1], transformation
    coor = torch.zeros(size=[h,w,l,dim+1]).to(device)
    coor[:,:,:,:dim] = grid
    coor[:,:,:,dim:] = 1
    trans_coor = torch.matmul(coor[:,:,:,np.newaxis,:],rigid_matrix.transpose(0,1))[:,:,:,0,:]
    # construct deformation field
    deform = trans_coor[:,:,:,:dim]
    deform = deform - grid
    deform = deform.permute(3,0,1,2)
    return deform

def get_deform_from_rigid_matrix_batch(rigid_matrix, shape):
    '''
    Parameters:
        rigid_matrix: shape [n,4,4], homo transform
        shape: shape [3], shape of image
    Returns:
        deform: shape [n,dim,h,w,l], output deformation field 
    '''
    h,w,l = shape
    dim = len(shape)
    device = rigid_matrix.device
    batch_size = rigid_matrix.shape[0]
    deform = torch.zeros(size=[batch_size,dim,h,w,l]).to(device)
    for batch_idx in range(0, batch_size):
        deform[batch_idx] = get_deform_from_rigid_matrix(rigid_matrix[batch_idx], shape)
    return deform

def spatial_transform(image, deform, mode="bilinear", padding_mode="zeros"):
    '''
    Parameters:
        image: tensor, shape [n,c,h,w,l], image to be transform
        deform: tensor, shape [n,dim,h,w,l], deformation field
    Returns:
        trans_image: shape [n,c,h,w,l], transformed image
    '''
    image = torch.unsqueeze(image, dim=0)
    deform = torch.unsqueeze(deform, dim=0)
    shape = image.shape[2:]
    return spatial_transform_batch(image, deform, mode=mode, padding_mode=padding_mode)[0]

def spatial_transform_batch(image, deform, mode="bilinear", padding_mode="zeros"):
    '''
    Parameters:
        image: shape [n,c,h,w,l], image to be transform
        deform: shape [n,dim,h,w,l], deformation field
    Returns:
        trans_image: shape [n,c,h,w,l], transformed image
    '''
    shape = image.shape[2:]
    grid = get_grid_tensor(shape).to(image.device)
    new_locs = grid + deform
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
    
    # adapt to nn.functional.grid_sample
    # move channels dim to last position
    # also not sure why, but the channels need to be reversed
    new_locs = new_locs.permute(0, 2, 3, 4, 1)
    new_locs = new_locs[..., [2, 1, 0]]
    return nnf.grid_sample(image, new_locs, align_corners=True, mode=mode, padding_mode=padding_mode)

def get_points_from_seg(seg):
    """
    Convert a input binary segment to a points in it.

    Parameters:
        seg (tensor): shape [1, H, W, L], voxels with value!=0 are valid.
    Returns:
        points_batch (tensor): shape [M, 3], M is the number of points.
    """
    _, h, w, l = seg.shape
    select_seg = seg>0.1
    grid_tensor = get_grid_tensor_by_param(h=h, w=w, l=l)
    grid_hs = grid_tensor[0:1, :, :, :]
    grid_ws = grid_tensor[1:2, :, :, :]
    grid_ls = grid_tensor[2:3, :, :, :]
    points_hs = grid_hs[select_seg]
    points_ws = grid_ws[select_seg]
    points_ls = grid_ls[select_seg]
    points = torch.stack([points_hs, points_ws, points_ls]).transpose(0,1)
    return points

def spatial_transform_by_param_tensor4D(image,
                                       trans_x=0,
                                       trans_y=0,
                                       trans_z=0,
                                       roll=0,
                                       pitch=0,
                                       yaw=0,
                                       center_x=0,
                                       center_y=0,
                                       center_z=0,
                                       mode="bilinear",
                                       padding_mode="zeros",
                                       ret_matrix=False,
                                       debug=False):
    """
        available mode: bilinear, nearest, bicubic
    """
    from .tensor_op import get_rigid_matrix_from_param, get_deform_from_rigid_matrix, spatial_transform
    # print(f"test spatial_transform trans_x:{trans_x} trans_y:{trans_y} trans_z:{trans_z} roll:{roll} pitch:{pitch} yaw:{yaw} center_x:{center_x} center_y:{center_y} center_z:{center_z}")
    tensor_image = image
    device = tensor_image.device
    # print(f"test shape image:{image.shape}")
    rigid_matrix = get_rigid_matrix_from_param(trans_x, trans_y, trans_z, roll, pitch, yaw, center_x, center_y, center_z).to(device)
    if(debug):
        print(f"[Debug] Spatial transform by param rigid_matrix:\n{rigid_matrix}")
    inv_rigid_matrix = get_inv_rigid_matrix(rigid_matrix).to(device)
    deform = get_deform_from_rigid_matrix(inv_rigid_matrix, shape=image.shape[1:]).to(device)
    # print(f"test device rigid_matrix:{rigid_matrix.device} inv_rigid_matrix:{inv_rigid_matrix.device} deform:{deform.device}")
    trans_tensor_image = spatial_transform(tensor_image, deform, mode=mode, padding_mode=padding_mode).to(device)
    trans_image = trans_tensor_image
    if(ret_matrix):
        return trans_image, rigid_matrix
    return trans_image
        
def get_points_from_seg_batch(seg_batch):
    """
    Convert a input binary segment to a points in it.

    Parameters:
        seg_batch (tensor): shape [N, 1, H, W, L], voxels with value!=0 are valid.
    Returns:
        points_batch (list): a list with points tensor as list item. Each points tensor is of shape [M, 3], M is the number of points.
    """
    points_batch = []
    batch_size = seg_batch.shape[0]
    for batch_idx in range(0, batch_size):
        points = get_points_from_seg(seg_batch[batch_idx])
        points_batch.append(points)
    return points_batch

def get_local_patch(x,mask,patch_h=64,patch_w=64,patch_l=64, margin=3):
    #get local patch of the binary mask
    #input [c,h,w,l] tensor
    _,h,w,l=mask.shape
    size_mask = get_mask_bbx4D_tensor(bin_mask=mask, pad_h=margin, pad_w=margin, pad_l=margin).to(mask.device)
    if(torch.sum(size_mask)!=0):
        _,hs,ws,ls=torch.where(size_mask!=0)
        min_h,max_h=torch.min(hs),torch.max(hs)
        min_w,max_w=torch.min(ws),torch.max(ws)
        min_l,max_l=torch.min(ls),torch.max(ls)
        start_h=min_h
        start_w=min_w
        start_l=min_l
        range_h = min(max_h - min_h + 1, patch_h)
        range_w = min(max_w - min_w + 1, patch_w)
        range_l = min(max_l - min_l + 1, patch_l)
    else:
        range_h = torch.randint(low=patch_h//4, high=patch_h//4*3)
        range_w = torch.randint(low=patch_w//4, high=patch_w//4*3)
        range_l = torch.randint(low=patch_l//4, high=patch_l//4*3)
        start_h = h//2 - range_h//2
        start_w = w//2 - range_w//2
        start_l = l//2 - range_l//2

    patch_x = torch.zeros(size=[x.shape[0], patch_h, patch_w, patch_l]).to(x.device)
    patch_mask = torch.zeros(size=[x.shape[0], patch_h, patch_w, patch_l]).to(size_mask.device)

    patch_x[:, :range_h, :range_w, :range_l] = \
         x[:,start_h:start_h+range_h, start_w:start_w+range_w, start_l:start_l+range_l]
    patch_mask[:, :range_h, :range_w, :range_l] = \
        size_mask[:,start_h:start_h+range_h, start_w:start_w+range_w, start_l:start_l+range_l]
    return patch_x, patch_mask

def get_local_patch_batch(x,mask,patch_h=64,patch_w=64,patch_l=64):
    #get local patch of the binary mask
    #input [b,c,h,w,l] tensor
    b_num=x.shape[0]
    patch_x_list=[]
    patch_mask_list=[]
    for b_idx in range(0,b_num):
        patch_x,patch_mask=get_local_patch(x[b_idx],mask[b_idx],patch_h=patch_h,patch_w=patch_w,patch_l=patch_l)
        patch_x_list.append(patch_x)
        patch_mask_list.append(patch_mask)
    patch_x_batch=torch.stack(patch_x_list)
    patch_mask_batch=torch.stack(patch_mask_list)
    return patch_x_batch,patch_mask_batch

def get_register_tensor(input_x,ref_x,default_value=0):
    batch_size=input_x.shape[0]
    trans_x=torch.zeros_like(input_x)
    for batch_idx in range(0,batch_size):
        fix_cube=ref_x[batch_idx,0,:,:,:].detach().cpu().numpy()
        fix_image=itk.GetImageFromArray(fix_cube)
        mov_cube=input_x[batch_idx,0,:,:,:].detach().cpu().numpy()
        mov_image=itk.GetImageFromArray(mov_cube)
        init_transform = itk.CenteredTransformInitializer(fix_image, mov_image, itk.Euler3DTransform(), itk.CenteredTransformInitializerFilter.GEOMETRY)
        regis=itk.ImageRegistrationMethod()
        regis.SetMetricAsMeanSquares()
        regis.SetInterpolator(itk.sitkLinear)
        regis.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,minStep=1e-2,numberOfIterations=100)
        regis.SetOptimizerScalesFromPhysicalShift()
        regis.SetInitialTransform(init_transform)
        transform=regis.Execute(fixed=fix_image,moving=mov_image)
        trans_image=itk.Resample(mov_image,fix_image,transform,itk.sitkLinear,default_value,mov_image.GetPixelID())
        trans_cube=itk.GetArrayFromImage(trans_image)
        trans_x[batch_idx,0,:,:,:]=torch.FloatTensor(trans_cube).to(input_x.device)
    trans_x.requires_grad=False
    return trans_x

def statistic_model_param(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def exam_tensor(tensor, name="default"):
    print(f"tensor {name}: shape:{tensor.shape} dtype:{tensor.dtype} max:{torch.max(tensor)} min:{torch.min(tensor)}")

def get_centroid_tensor4D(tensor, round=False):
    _, hs, ws, ls = torch.where(tensor>1e-2)
    mean_h = torch.mean(hs.float())
    mean_w = torch.mean(ws.float())
    mean_l = torch.mean(ls.float())
    if(round):
        mean_h = int(torch.round(mean_h))
        mean_w = int(torch.round(mean_w))
        mean_l = int(torch.round(mean_l))
    return torch.FloatTensor([mean_h, mean_w, mean_l])

def center_align_by_seg_tensor4D(
    fix_seg,
    mov_seg,
    mov_image=None,
    jit_range=[0,0,0]
):
    fix_seg_centroid = get_centroid_tensor4D(fix_seg)
    mov_seg_centroid= get_centroid_tensor4D(mov_seg)
    trans_h, trans_w, trans_l = fix_seg_centroid - mov_seg_centroid
    # jit
    # h
    jit_range_h = jit_range[0]
    jit_h = torch.randint(low=-jit_range_h, high=jit_range_h+1, size=[])
    trans_h += jit_h
    # w
    jit_range_w = jit_range[1]
    jit_w = torch.randint(low=-jit_range_w, high=jit_range_w+1, size=[])
    trans_w += jit_w
    # l
    jit_range_l = jit_range[2]
    jit_l = torch.randint(low=-jit_range_l, high=jit_range_l+1, size=[])
    trans_l += jit_l

    output_seg = spatial_transform_by_param_tensor4D(
        image = mov_seg,
        trans_x = trans_h,
        trans_y = trans_w,
        trans_z = trans_l,
        mode = "nearest"
    )
        
    output_image = None
    if(mov_image is not None):
        output_image = spatial_transform_by_param_tensor4D(
            image = mov_image,
            trans_x = trans_h,
            trans_y = trans_w,
            trans_z = trans_l,
            mode = "bilinear"
        )

    align_dict = {
        "trans_x" : trans_h,
        "trans_y" : trans_w,
        "trans_z" : trans_l,
        "output_seg" : output_seg,
        "output_image" : output_image
    }
    return align_dict

def center_align_by_seg_tensor4D_batch(
    fix_seg_batch,
    mov_seg_batch,
    mov_image_batch,
    jit_range=[0, 0, 0]
):
    batch_num = fix_seg_batch.shape[0]
    output_seg_list = []
    output_image_list = []
    for batch_idx in range(0, batch_num):
        fix_seg = fix_seg_batch[batch_idx]
        mov_seg = mov_seg_batch[batch_idx]
        mov_image = mov_image_batch[batch_idx]
        cur_align_dict = center_align_by_seg_tensor4D(
            fix_seg=fix_seg,
            mov_seg=mov_seg,
            mov_image=mov_image,
            jit_range=jit_range
        )
        output_seg_list.append(cur_align_dict["output_seg"])
        output_image_list.append(cur_align_dict["output_image"])
    output_seg_batch = torch.stack(output_seg_list, dim=0)
    output_image_batch = torch.stack(output_image_list, dim=0)
    align_dict = {
        "output_seg" : output_seg_batch,
        "output_image" : output_image_batch
    }
    return align_dict

def cal_ahd_forward_general3D(
    seg_src,
    seg_dst,
    scale=1.0,
    print_msg_flag=True,
    thresh_eps=1e-1
):
    """
    notes:
        1. calculate ahd from 'seg_src' to 'seg_dst'
        2. Input are tensor
    """
    if(print_msg_flag):
        print("Calculate Ahd Forward")
    # src pts
    src_pts = torch.where(torch.FloatTensor(seg_src>thresh_eps))
    src_pts_tensor = torch.transpose(torch.stack(src_pts),1,0)
    # dst pts
    dst_pts = torch.where(torch.FloatTensor(seg_dst>thresh_eps))
    dst_pts_tensor = torch.transpose(torch.stack(dst_pts),1,0)
    # dist matrix
    dist_matrix = torch.sqrt(torch.sum((src_pts_tensor[:,np.newaxis,:] - dst_pts_tensor[np.newaxis,:,:])**2, dim=2))
    dist_min = torch.min(dist_matrix, dim=1).values
    ahd = torch.mean(dist_min).numpy()*scale
    return ahd

def cal_ahd_backward_general3D(
    seg_src,
    seg_dst,
    scale=1.0,
    print_msg_flag=True,
    thresh_eps=1e-1
):
    """
    notes:
        calculate ahd from 'seg_dst' to 'seg_src'
    """
    if(print_msg_flag):
        print("Calculate Ahd Backward")
    cur_seg_src = seg_src
    cur_seg_dst = seg_dst
    ahd = cal_ahd_forward_general3D(
        seg_src=cur_seg_dst,
        seg_dst=cur_seg_src,
        scale=scale,
        print_msg_flag=False,
        thresh_eps=thresh_eps
    )
    return ahd

def cal_ahd_symmetric_general3D(
    seg_a,
    seg_b,
    scale=1.0,
    print_msg_flag=True,
    thresh_eps=1e-1
):
    """
    notes:
        calculate ahd symmetrically, average result from 'cal_ahd_forward' and 'cal_ahd_backward'
    """
    if(print_msg_flag):
        print("Calculate Ahd Symmetric")
    # forward
    ahd_forward = cal_ahd_forward_general3D(
        seg_src=seg_a,
        seg_dst=seg_b,
        scale=scale,
        print_msg_flag=False,
        thresh_eps=thresh_eps
    )
    # backward
    ahd_backward = cal_ahd_backward_general3D(
        seg_src=seg_a,
        seg_dst=seg_b,
        scale=scale,
        print_msg_flag=False,
        thresh_eps=thresh_eps
    )
    # symmetric
    ahd_symmetric = (ahd_forward + ahd_backward)/2
    return ahd_symmetric

def cal_ahd_forward_lowmem_general3D(
    seg_src,
    seg_dst,
    scale=1.0,
    block_size=10000,
    print_msg_flag=True,
    thresh_eps=1e-1,
    use_cuda=False,
    debug=False
):
    """
    notes:
        1. calculate ahd from 'seg_src' to 'seg_dst'
        2. Use block strategy to speed up and save memory
    """
    if(print_msg_flag):
        print("Calculate Ahd Forward")
    # src pts
    src_pts = torch.where(torch.FloatTensor(seg_src>thresh_eps))
    src_pts_tensor = torch.transpose(torch.stack(src_pts),1,0)
    if(use_cuda):
        src_pts_tensor = src_pts_tensor.cuda()
    src_block_num = math.ceil(src_pts_tensor.shape[0]/block_size)
    # dst pts
    dst_pts = torch.where(torch.FloatTensor(seg_dst>thresh_eps))
    dst_pts_tensor = torch.transpose(torch.stack(dst_pts),1,0)
    if(use_cuda):
        dst_pts_tensor = dst_pts_tensor.cuda()
    dst_block_num = math.ceil(dst_pts_tensor.shape[0]/block_size)
    
    # init min vector
    dist_min_vector = torch.ones(size=[src_pts_tensor.shape[0]])*100000000
    # traverse src_pts by block
    for src_block_idx in range(0, src_block_num):
        src_start_idx = src_block_idx * block_size
        src_end_idx = (src_block_idx + 1) * block_size
        block_src_pts_tensor = src_pts_tensor[src_start_idx:src_end_idx, :]
        slice_min_vector = None
        # traverse dst_pts by block
        for dst_block_idx in range(0, dst_block_num):
            dst_start_idx = dst_block_idx * block_size
            dst_end_idx = (dst_block_idx + 1) * block_size
            block_dst_pts_tensor = dst_pts_tensor[dst_start_idx:dst_end_idx, :]
            # calculate block dist matrix
            block_dist_matrix = torch.sqrt(
                    torch.sum((block_src_pts_tensor[:,np.newaxis,:] - block_dst_pts_tensor[np.newaxis,:,:])**2, dim=2)
                )
            block_min_vector = torch.min(block_dist_matrix, dim=1).values
            # save min value from block_min_vector into slice_min_vector
            if(slice_min_vector is None):
                slice_min_vector = block_min_vector
            else:
                slice_min_vector = torch.min(slice_min_vector, block_min_vector)
            if(debug):
                print(f"handling src_start_idx:{src_start_idx} src_end_idx:{src_end_idx} dst_start_idx:{dst_start_idx} dst_end_idx:{dst_end_idx} block_src_pts_tensor:{block_src_pts_tensor.shape} block_dst_pts_tensor:{block_dst_pts_tensor.shape} block_dist_matrix:{block_dist_matrix.shape} block_min_vector:{block_min_vector.shape} slice_min_vector:{slice_min_vector.shape}\n")
        # integrate slice_min_vector into dist_min_vector
        dist_min_vector[src_start_idx:src_end_idx] = slice_min_vector
    
    # calculate ahd
    ahd = torch.mean(dist_min_vector).cpu().numpy()*scale
    return ahd

def cal_ahd_backward_lowmem_general3D(
    seg_src,
    seg_dst,
    scale=1.0,
    block_size=10000,
    print_msg_flag=True,
    thresh_eps=1e-1,
    use_cuda=False,
    debug=False
):
    """
    notes:
        1. calculate ahd from 'seg_src' to 'seg_dst'
        2. Use block strategy to speed up and save memory
    """
    if(print_msg_flag):
        print("Calculate Ahd Backward")
    
    cur_seg_src = seg_src
    cur_seg_dst = seg_dst
    ahd = cal_ahd_forward_lowmem_general3D(
        seg_src=cur_seg_dst,
        seg_dst=cur_seg_src,
        scale=scale,
        block_size=block_size,
        print_msg_flag=False,
        thresh_eps=thresh_eps,
        use_cuda=use_cuda,
        debug=debug
    )
    return ahd

def cal_ahd_symmetric_lowmem_general3D(
    seg_a,
    seg_b,
    scale=1.0,
    block_size=10000,
    print_msg_flag=True,
    thresh_eps=1e-1,
    use_cuda=False,
    debug=False
):
    if(print_msg_flag):
        print("Calculate Ahd Symmetric")
    # forward
    ahd_forward = cal_ahd_forward_lowmem_general3D(
        seg_src=seg_a,
        seg_dst=seg_b,
        scale=scale,
        block_size=block_size,
        print_msg_flag=False,
        thresh_eps=thresh_eps,
        use_cuda=use_cuda,
        debug=debug
    )
    # backward
    ahd_backward = cal_ahd_backward_lowmem_general3D(
        seg_src=seg_a,
        seg_dst=seg_b,
        scale=scale,
        block_size=block_size,
        print_msg_flag=False,
        thresh_eps=thresh_eps,
        use_cuda=use_cuda,
        debug=debug
    )
    # symmetric    
    ahd_symmetric = (ahd_forward + ahd_backward)/2
    return ahd_symmetric