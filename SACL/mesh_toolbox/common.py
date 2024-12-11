# common operation code
import shutil
import sys
import json

from .dependencies import *
import time
from scipy.spatial.transform import Rotation
from multiprocessing import Pool
from tqdm import tqdm

# path functions
def get_mark(mha_path):
    name=os.path.basename(mha_path)
    mark=name[:name.index(".")]
    return mark

def get_name(mha_path):
    file_name=os.path.basename(mha_path)
    name=file_name[:file_name.index(".")]
    return name

def get_name_list(path_list):
    name_list = []
    for path in path_list:
        name = get_name(path)
        name_list.append(name)
    return name_list

def get_suffix(mha_path):
    file_name=os.path.basename(mha_path)
    suffix=file_name[file_name.index(".")+1:]
    return suffix

def make_dir(path, exists_ok=True):
    os.makedirs(path, exist_ok=exists_ok)

def make_dir_list(path_list, exists_ok=True):
    for path in path_list:
        make_dir(path=path, exists_ok=exists_ok)

def make_parent_dir(path,exist_ok=True):
    os.makedirs(os.path.dirname(path),exist_ok=exist_ok)

def make_parent_dir_list(path_list,exist_ok=True):
    for path in path_list:
        make_parent_dir(path)

def get_sub_dirs(current_dir):
    sub_dir_list=[]
    for sub_path in glob.glob(f"{current_dir}/*"):
        if(os.path.isdir(sub_path)):
            sub_dir_list.append(sub_path)
    return sub_dir_list

# print functions
def print_dict(data_dict, name=""):
    msg = name
    for key in data_dict.keys():
        value = data_dict[key]
        msg += f" {key}:{value}"
    return msg

# log functions
def init_log(log_path):
    for logger_name in ["Model","Dataset","Train"]:
        logger=logging.getLogger(logger_name)
        if(len(logger.handlers)==0):
            file_handler=logging.FileHandler(filename=log_path,mode="a")
            stream_handler = logging.StreamHandler()
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        
def log(msg, add_time=True):
    logger=logging.getLogger("Train")
    log_msg = msg
    if(add_time):
        log_msg = f"[{get_cur_time_str()}] {log_msg}"
    logger.info(log_msg)

class MultiProcess_Processor:
    def __init__(self, process_num=10):
        self.process_num = process_num
    
    def test_func(self, process_func, data_list, check_data_list=False):
        print("[Test func]")
        if(check_data_list):
            print("[Check data_list]")
            print(data_list[0])
        return process_func(data_list[0])
    
    def exec_func(self, process_func, data_list, check_data_list=False):
        """

        Args:
            func : the function to be execute, notes that the args of the func should be a datalist
            data_list : datalist to be execute
        """
        

        print("[MultiProcess Processor]")
        data_num = len(data_list)
        if(check_data_list):
            print("[Check data_list]")
            for data_idx, data in enumerate(data_list):
                print(f"data idx:{data_idx} data:{data}")
            print("")
        print(f"process num:{self.process_num} data_num:{data_num}")
        process_pool = Pool(processes=self.process_num)
        # try 
        ret_list = []
        try:
            result_list = []
            for data_idx in range(0, len(data_list)):
                result_list.append(process_pool.apply_async(process_func, data_list[data_idx:data_idx+1]))
            print("[Waiting Subprocess]")       
            for res_idx in tqdm(range(0, len(result_list))):
                ret_value = result_list[res_idx].get()
                print(f"Task {res_idx}/{data_num} return: ", end="")
                print(ret_value)
                ret_list.append(ret_value)
        # handle ctrl+c sigint
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            process_pool.terminate()
            sys.exit(1)
        else:        
            process_pool.close()
            process_pool.join()
        print("[Subprocess finished]")
        return ret_list

def single_process_exec(process_func, process_num, data_list, check_data_list=False):
    ret_list = []
    for data in tqdm(data_list):
        ret = process_func(data)
        ret_list.append(ret)
    return ret_list

def multi_process_exec(process_func, process_num, data_list, check_data_list=False):
    processor = MultiProcess_Processor(process_num=process_num)
    return processor.exec_func(process_func=process_func, data_list=data_list, check_data_list=check_data_list)

def common_json_dump(json_object, file_path):
    json_file = open(file_path, mode="w")
    json.dump(json_object, json_file)
    json_file.close()

def common_json_load(file_path):
    json_file = open(file_path, mode="r")
    json_object = json.load(json_file)
    json_file.close()
    return json_object

# sample
def not_putback_sampling(sample_list,sample_num,ret_ids=False):
    choice_list=[]
    ids_list=[]
    for i in range(sample_num):
        choice_list.append(random.choice(sample_list))
        ids_list.append(sample_list.index(choice_list[-1]))
        sample_list.remove(choice_list[-1])
    if(ret_ids):
        return choice_list,ids_list
    else:
        return choice_list

# spliter
def dataset_split_func(data_paths,
                       split_subset_name_list=["train", "val", "test"],
                       split_subset_num_list=[15, 5, 10]
                       ):
    import copy
    left_data_paths = copy.copy(data_paths)
    assert len(split_subset_name_list)==len(split_subset_num_list)
    assert len(left_data_paths)>=sum(split_subset_num_list)
    split_subset_paths_list = []
    for subset_name, subset_num in zip(split_subset_name_list, split_subset_num_list):
        subset_paths = not_putback_sampling(sample_list=left_data_paths, sample_num=subset_num)
        split_subset_paths_list.append(subset_paths)
        print(f"splitting {subset_name}: {len(subset_paths)}")
    return split_subset_paths_list
    
# repeater
from itertools import repeat
def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def filter_func(path_list, filter_list):
    filter_path_list = []
    for path in path_list:
        select_flag = False
        for filter_name in filter_list:
            if(filter_name in path):
                select_flag = True
                break
        if(select_flag):
            filter_path_list.append(path)
    print(f"path num before filter:{len(path_list)} after_filter:{len(filter_path_list)}")
    return filter_path_list

# time
def get_cur_time_str(format="%Y-%m-%d %H:%M:%S"):
    cur_time = time.localtime()
    cur_time_str = time.strftime(format, cur_time)
    return cur_time_str

def exam_path(path):
    abs_path = os.path.abspath(path)
    print(f"abs_path:{abs_path} exist:{os.path.exists(abs_path)}")
    return True

def file_copy_func(src_path, dst_path, make_dirs=True):
    if(make_dirs):
        make_parent_dir(dst_path)
    shutil.copy(src_path, dst_path)

def round_int(x):
    return int(round(x))

def get_clip_value_list(value_list, clip_min, clip_max):
    value_array = np.array(value_list)
    clip_value_array = np.clip(value_array, a_min=clip_min, a_max=clip_max)
    clip_value_list = clip_value_array.tolist()
    return clip_value_list