from .dependencies import *
from scipy.spatial.transform import Rotation

# math functions
def rad(angle):
    return angle/180*np.pi

def degree(rad):
    return rad/np.pi*180

# common geometry ops
def get_euler_from_rotation_matrix(rot_matrix, use_rad=True):
    use_degree = not use_rad
    matrix = copy.deepcopy(rot_matrix)
    rotation = Rotation.from_matrix(matrix[:3,:3])
    euler = rotation.as_euler(seq="zyx", degrees=use_degree)
    roll = -euler[2]
    pitch = -euler[1]
    yaw = -euler[0]
    return roll, pitch, yaw

def get_param_from_rigid_matrix(rigid_matrix,
                                center_x=0,
                                center_y=0,
                                center_z=0,
                                use_rad=True,
                                debug=False):
    matrix = copy.deepcopy(rigid_matrix)
    center = np.array([center_x, center_y, center_z])
    # trans
    trans = matrix[:3, 3]
    center = np.array(center)
    trans = trans - center + matrix[:3,:3]@center
    trans_x = trans[0]
    trans_y = trans[1]
    trans_z = trans[2]
    # eulers
    rot_matrix = matrix[:3,:3]
    eulers = get_euler_from_rotation_matrix(rot_matrix, use_rad=use_rad)
    roll = eulers[0]
    pitch = eulers[1]
    yaw = eulers[2]
    if(debug):
        print("")
        print(f"test get_param_from_rigid_matrix")
        print(f"type:{type(center)} center:{center}")
        print(f"origin matrix:\n{rigid_matrix}")
        compose_matrix = get_rigid_matrix_from_param_array(
            trans_x = trans_x,
            trans_y = trans_y,
            trans_z = trans_z,
            roll = roll/180*np.pi,
            pitch = torch.FloatTensor([pitch/180*np.pi]),
            yaw = torch.FloatTensor([yaw/180*np.pi]),
            center_x = float(center[0]),
            center_y = float(center[1]),
            center_z = float(center[2])
        )
        print(f"compose matrix:\n{compose_matrix}")
    # param dict
    param_dict = {
        "trans_x" : trans_x,
        "trans_y" : trans_y,
        "trans_z" : trans_z,
        "roll" : roll,
        "pitch" : pitch,
        "yaw" : yaw,
        "matrix" : matrix.tolist()
    }
    return param_dict


def get_rigid_matrix_from_param_array(trans_x,trans_y,trans_z,roll,pitch,yaw,center_x=63.5,center_y=63.5,center_z=63.5,debug=False):
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
    from math import cos,sin
    # center
    Rc = np.zeros(shape=[4,4])
    Rc[0,0] = 1
    Rc[1,1] = 1
    Rc[2,2] = 1
    Rc[0,3] = -center_x
    Rc[1,3] = -center_y
    Rc[2,3] = -center_z
    Rc[3,3] = 1
    
    # rotation
    # Rx
    Rx = np.zeros(shape=[4,4])
    Rx[0,0] = 1
    Rx[1,1] = cos(roll)
    Rx[1,2] = sin(roll)
    Rx[2,1] = -sin(roll)
    Rx[2,2] = cos(roll)
    Rx[3,3] = 1
    # Ry
    Ry = np.zeros(shape=[4,4])
    Ry[0,0] = cos(pitch)
    Ry[0,2] = -sin(pitch)
    Ry[1,1] = 1
    Ry[2,0] = sin(pitch)
    Ry[2,2] = cos(pitch)
    Ry[3,3] = 1
    # Rz
    Rz = np.zeros(shape=[4,4])
    Rz[0,0] = cos(yaw)
    Rz[0,1] =sin(yaw)
    Rz[1,0] = -sin(yaw)
    Rz[1,1] = cos(yaw)
    Rz[2,2] = 1
    Rz[3,3] = 1
    # combine
    Rr = Rx@Ry@Rz
    
    # transformation
    Rt = np.zeros(shape=[4,4])
    Rt[0,0] = 1
    Rt[1,1] = 1
    Rt[2,2] = 1
    Rt[0,3] = trans_x
    Rt[1,3] = trans_y
    Rt[2,3] = trans_z
    Rt[3,3] = 1

    # de-center
    Rdc = np.zeros(shape=[4,4])
    Rdc[0,0] = 1
    Rdc[1,1] = 1
    Rdc[2,2] = 1
    Rdc[0,3] = center_x
    Rdc[1,3] = center_y
    Rdc[2,3] = center_z
    Rdc[3,3] = 1
        
    # total
    R = Rt@Rdc@Rr@Rc
    
    if(debug):
        print(f"[Debug] get_rigid_matrix_from_param: ")
        print(f"trans_x:{trans_x} trans_y:{trans_y} trans_z:{trans_z} roll:{roll} pitch:{pitch} yaw:{yaw} center_x:{center_x} center_y:{center_y} center_z:{center_z}\n")
        print(f"result matrix:\n{R}\n")
    return R