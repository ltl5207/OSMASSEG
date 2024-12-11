from .dependencies import *
from .common import *

intp_dict = {
    "nn":cv2.INTER_NEAREST,
    "cubic":cv2.INTER_CUBIC,
    "linear":cv2.INTER_LINEAR
}

# read and write float gary images
def read_image_uint8_gray(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def write_image_uint8_gray(image, image_path, shape=None, interpolation="cubic"):
    if(shape is not None):
        shape=tuple(shape)
        image=cv2.resize(image,dsize=shape, interpolation=intp_dict[interpolation])
    cv2.imwrite(image_path, image)

def read_image_float_gray(image_path, max_value=255.0):
    image = read_image_uint8_gray(image_path)
    image = cvt_image_uint8_to_float(image, max_value=max_value)
    return image

def write_image_float_gray(image, image_path, max_value=255.0, shape=None, interpolation="cubic"):
    image = cvt_image_float_to_uint8(image, max_value=max_value)
    return write_image_uint8_gray(image, image_path, shape, interpolation=intp_dict[interpolation])

def write_image_direct_gray(image, image_path):
    cv2.imwrite(image_path, image)

# read and write float color images
def read_image_uint8_color(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def write_image_uint8_color(image, image_path, shape=None, interpolation="cubic"):
    if(shape is not None):
        shape=tuple(shape)
        image=cv2.resize(image,dsize=[shape[1], shape[0]], interpolation=intp_dict[interpolation])
    cv2.imwrite(image_path, image)

def read_image_float_color(image_path, max_value=255.0):
    image = read_image_uint8_color(image_path)
    image = cvt_image_uint8_to_float(image, max_value=max_value)
    return image

def write_image_float_color(image, image_path, max_value=255.0, shape=None, interpolation="cubic"):
    make_parent_dir(image_path)
    image = cvt_image_float_to_uint8(image, max_value=max_value)
    return write_image_uint8_color(image, image_path, shape, interpolation=interpolation)

# image convert
def cvt_image_float_to_uint8(image, max_value=255):
    image = np.clip(image*max_value, 0, max_value).astype(np.uint8)
    return image

def cvt_image_uint8_to_float(image, max_value=255.0):
    image = np.clip(image.astype(np.float32)/max_value, 0, 1)
    return image

def cvt_image_gray_to_color(image):
    return cv2.cvtColor(src=image, code=cv2.COLOR_GRAY2BGR)

def cvt_image_color_to_gray(image):
    return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

def crop_image_array2D(image, center, size, hw_ratio=None):
    """
    Crop 2D image slice based on center and size
    image: 2D array, [h, w, c]
    center: [center_h, center_w]
    size: [size_h, size_w]
    """
    image_h, image_w = image.shape[0], image.shape[1]
    center_h, center_w = center
    size_h, size_w = size
    if(hw_ratio is not None):
        size_w = round_int(size_h/hw_ratio)
    start_h = round_int(max(center_h - size_h//2, 0))
    end_h = round_int(min(start_h + size_h, image_h-1))
    start_w = round_int(max(center_w - size_w//2, 0))
    end_w = round_int(min(start_w + size_w, image_w-1))
    crop_image = image[start_h:end_h, start_w:end_w]
    return crop_image

# color convert
def cvt_image_bgr_to_rgb(image_array):
    # color array is of shape[3]
    cvt_image = image_array.copy()
    cvt_image[:, :, 0] = image_array[:, :, 2]
    cvt_image[:, :, 2] = image_array[:, :, 0]
    return cvt_image

def cvt_image_rgb_to_bgr(image_array):
    # color array is of shape[3]
    cvt_image = image_array.copy()
    cvt_image[:, :, 0] = image_array[:, :, 2]
    cvt_image[:, :, 2] = image_array[:, :, 0]
    return cvt_image

def cvt_color_bgr_to_rgb(color_array):
    # color array is of shape[3]
    cvt_color = color_array.copy()
    cvt_color[0] = color_array[2]
    cvt_color[2] = color_array[0]
    return cvt_color

def cvt_color_rgb_to_bgr(color_array):
    # color array is of shape[3]
    cvt_color = color_array.copy()
    cvt_color[0] = color_array[2]
    cvt_color[2] = color_array[0]
    return cvt_color

def blend_image_with_color_mask(image,mask,blend_color,alpha=0.5):
    # mask [h,w] or [h,w,c] float [0,1]
    # image [h,w,c] float [0,1]
    if(len(mask.shape)!=len(image.shape)):
        c = image.shape[2]
        blend_mask = np.repeat(mask[:,:, np.newaxis], repeats=c, axis=2)
    else:
        blend_mask = mask
    blend_color = blend_mask*blend_color
    blend_image=image*(1-blend_mask)+image*blend_mask*(1-alpha)+blend_color*blend_mask*alpha
    return blend_image

def blend_image_with_image(image_a, image_b, color_a=None, color_b=None, alpha_color_a=0.5, alpha_color_b=0.5, alpha=0.5):
    # image_a, image_b [h,w,c] float [0,1]
    # color_a, color_b [3] float [0,1]
    if(color_a is None):
        colored_image_a = image_a
    else:
        mask_a = np.ones_like(image_a)
        colored_image_a = blend_image_with_color_mask(image=image_a, mask=mask_a, blend_color=color_a, alpha=alpha_color_a)
    
    if(color_b is None):
        colored_image_b = image_b
    else:
        mask_b = np.ones_like(image_b)
        colored_image_b = blend_image_with_color_mask(image=image_b, mask=mask_b, blend_color=color_b, alpha=alpha_color_b)
    
    blend_image = colored_image_a*(1-alpha) + colored_image_b*(alpha)
    return blend_image

def blend_image_float_gray_by_plt_cm(image, cm_type):
    from matplotlib import cm
    plt_cm = cm.get_cmap(cm_type)
    blend_image = plt_cm(image)
    c = blend_image.shape[2]
    blend_image = blend_image[:, :, :c-1] * blend_image[:, :, c-1:]
    blend_image = cvt_image_rgb_to_bgr(blend_image)
    return blend_image

def get_cv2contours_binary_image_float(binary_image):
    """
    input: binary seg float image
    return: cv2_Contours
    """
    cv2_contours, _ = cv2.findContours(image=cvt_image_float_to_uint8(binary_image),mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    return cv2_contours    

def draw_cv2contours_image_float(image, contours, color):
    """
    input: float image, cv2_Contours, color [3] in [0,1]
    return: image with contour blended
    """
    if(color is not None):
        color = np.array(color)*255.0
    image = cvt_image_float_to_uint8(image)
    image = cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=color)
    return cvt_image_uint8_to_float(image)

def get_contours_mask_binary_image_float(binary_image):
    cur_binary_image = binary_image
    cv2_contours = get_cv2contours_binary_image_float(
        binary_image=cur_binary_image
    )
    shape_h = binary_image.shape[0]
    shape_w = binary_image.shape[1]
    contours_color_mask = np.zeros(shape=[shape_h, shape_w, 3])
    contours_color_mask = draw_cv2contours_image_float(
        image = contours_color_mask,
        contours = cv2_contours,
        color = [0, 1, 0]
    )
    contours_binary_mask = np.max(contours_color_mask, axis=2)
    contours_binary_mask = np.where(contours_binary_mask>1e-1, 1, 0).astype(np.float32)
    return contours_binary_mask

class ImageGrider:
    def __init__(self, n_row, n_col, row_size):
        self.n_row = n_row
        self.n_col = n_col
        self.row_size = row_size
        self.path_matrix = {}
        for col_idx in range(0, self.n_col):
            self.path_matrix[col_idx] = {}
        self.image_matrix = {}
        for col_idx in range(0, self.n_col):
            self.image_matrix[col_idx] = {}
        self.col_index_list = [0]
    
    def set_path(self, row_idx, col_idx, image_path):
        self.path_matrix[col_idx][row_idx] = image_path
    
    def draw_grid(self, grid_path, print_info=False):
        # calculate index
        for col_idx in range(0, self.n_col):
            max_col_size = 0
            for row_idx in range(0, self.n_row):
                image_path = self.path_matrix[col_idx][row_idx]
                image = read_image_float_color(image_path)
                image_h, image_w, _ = image.shape
                zoom_scale = self.row_size/image_h
                zoom_h = self.row_size
                zoom_w = int(round(zoom_scale*image_w))
                zoom_image = cv2.resize(src=image, dsize=[zoom_w, zoom_h], interpolation=intp_dict["cubic"])
                max_col_size = max(max_col_size, zoom_w)
                self.image_matrix[col_idx][row_idx] = zoom_image
            cur_col_index = self.col_index_list[-1] + max_col_size
            self.col_index_list.append(cur_col_index)
        # set image
        grid_image_h = self.n_row * self.row_size
        grid_image_w = self.col_index_list[-1]
        grid_image = np.zeros(shape=[grid_image_h, grid_image_w, 3])
        for col_idx in range(0, self.n_col):
            for row_idx in range(0, self.n_row):
                image = self.image_matrix[col_idx][row_idx]
                src_row_index = row_idx * self.row_size
                dst_row_index = (row_idx + 1)*self.row_size
                src_col_index = self.col_index_list[col_idx]
                dst_col_index = self.col_index_list[col_idx+1]
                dst_col_index = min(dst_col_index, image.shape[1] + src_col_index)
                grid_image[src_row_index:dst_row_index, src_col_index:dst_col_index, :] = image
        make_parent_dir(grid_path)
        write_image_float_color(grid_image, grid_path)