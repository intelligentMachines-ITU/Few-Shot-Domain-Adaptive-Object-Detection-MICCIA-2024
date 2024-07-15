import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
##############################################################################33


def get_obj_count(dir):
    object_count = {}
    #get data for further preprocessing
    for magnification in os.listdir(dir): 
        ann_paths = os.listdir(os.path.join(dir,magnification))
        local_obj_count = {}
        for path in ann_paths:
            annotations_file = os.path.join(dir,magnification,path)
            annotations = np.loadtxt(annotations_file, ndmin=2, delimiter=" ")
            #get values for the class object counts at each resolutio
            for clas in annotations[:,0]:
                clas = int(clas)
                if clas in local_obj_count.keys():
                    local_obj_count[clas]+=1
                else:
                    local_obj_count[clas]=1
        #add the magnification wise details 
        object_count[magnification] = local_obj_count
    return object_count

def plot(data, dir):
    #categories = list(str(data.keys()))
    values = [data[0],data[1],data[2],data[3]]
    # Create a bar chart
    #plt.bar(['gametocyte', 'schizont', 'trophozoite', 'ring'],values )
    plt.bar(['Large Lymph', 'Neutrophil', 'Small Lymph', 'Monocyte'],values )
    for index, value in enumerate(values):
        plt.text(value, index,str(value))
    # Add labels and title
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Raabin Object count after augmentation')
    plt.savefig('rabin_obj_count_after_augmentation.png')
    # Show the plot
    #plt.show()
###########################################################################################      

if __name__ == '__main__':
    dir = 'datasets/raabin/rabin/data_visualization'
    #sor_dir = Path('datasets/Olympus_tar_aug/labels/train')
    sor_dir = Path('datasets/raabin/rabin/Rabin_tar_aug/labels')
    obj_count =  get_obj_count(sor_dir)
    print(obj_count)
    plot(obj_count['1000x'], dir)
    