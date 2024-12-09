import os
import shutil
import numpy as np

if __name__ == '__main__':
    # get the 9 patients' names
    print(os.getcwd())
    patients_namelist_raw = os.listdir('../data/train9/Seg/')
    patients_namelist = list(map(lambda x:x.split('_')[0], patients_namelist_raw))
    patients_namelist = np.unique(patients_namelist)

    patient_name = 'chenxi_'
    patient_filenames_seg = []
    patient_filenames_jpg = []
    for i,filename in enumerate(patients_namelist_raw):
        if filename.startswith(patient_name):
            #patient_filenames_seg.append(filename)
            shutil.copy(os.path.join('../data/train9/Seg', filename), '../data/train1/Seg/')
            #patient_filenames_jpg.append(filename.split('.')[0]+'.jpg')
            shutil.copy(os.path.join('../data/train9/JPEG', (filename.split('.')[0]+'.jpg')), '../data/train1/JPEG')

    #验证
    namelist1 = list(map(lambda x:x.split('.')[0], os.listdir('../data/train1/JPEG/')))
    namelist2 = list(map(lambda x: x.split('.')[0], os.listdir('../data/train1/Seg/')))
    nameset1 = set(namelist1)
    nameset2 = set(namelist2)
    print(nameset2 == nameset1)