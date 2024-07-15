import cv2
import os
import shutil
import random
import numpy as np
import math
from pathlib import Path

###############################################################################################

def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


def get_image(label):
    img = label.replace('labels','images')
    img =  img.replace('.txt', '.png')
    return img

"""
make all the required dirs
"""
def make_dirs(parent):
    #image dirs for olympus at each resolution and chucnk
    mkdir(parent/'Transform_olympus_aug'/'images'/'train'/'100x')
    mkdir(parent/'Transform_olympus_aug'/'images'/'train'/'400x')
    mkdir(parent/'Transform_olympus_aug'/'images'/'train'/'1000x')
    
    #label dirs for olympus at each resolution and chucnk
    mkdir(parent/'Transform_olympus_aug'/'labels'/'train'/'100x')
    mkdir(parent/'Transform_olympus_aug'/'labels'/'train'/'400x')
    mkdir(parent/'Transform_olympus_aug'/'labels'/'train'/'1000x')


    
def get_empty_rejions(indices):
    result_sets = []
    current_set = []
    for i in range(len(indices)-1):
        current_set.append(indices[i])
        if indices[i + 1] != indices[i] + 1:
            if len(current_set) == 80:
                result_sets.append(current_set.copy())
            current_set.clear()
        if len(current_set) == 80:
            result_sets.append(current_set.copy())
            current_set.clear()
    return result_sets


  
def get_empty_rejions_smaller_chunks(indices):
    result_sets = []
    current_set = []
    for i in range(len(indices)-1):
        current_set.append(indices[i])
        if indices[i + 1] != indices[i] + 1:
            if len(current_set) >= 80:
                result_sets.append(current_set.copy())
            current_set.clear()
    # Check the last set
    if len(current_set) >= 80:
        result_sets.append(current_set)
        
    return result_sets



def find_non_overlapping_position(annotations, image_shape):
    # Find non-overlapping position in the background
    empty_rejions = []
    bboxes = np.empty((annotations.shape), int)
    image_width,image_height = image_shape[0],image_shape[1]
    for i in range(len(annotations)):
        x_center = annotations[i][1]
        y_center = annotations[i][2]
        width = annotations[i][3]
        height = annotations[i][4]
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        bboxes[i] = np.array([[annotations[i][0],x_min, y_min, x_max, y_max]])
    mask = np.ones((image_shape[0],image_shape[1]))
    for box_coord in bboxes:
        mask[box_coord[1]:box_coord[3],:] = 0
    #plt.imshow(mask,cmap='gray')
    empty_row_indices = np.where(np.all(mask==1, axis=1))[0]
    empty_rejions = get_empty_rejions(empty_row_indices)
    #print(empty_rejions)
    return empty_rejions


def preprocess_data(dir):
    object_count = {}
    less_objects = {}
    more_objects = {}
    object_present = {}
    object_absent = {} 
    #get data for further preprocessing
    for magnification in os.listdir(dir): 
        ann_paths = os.listdir(os.path.join(dir,magnification))
        less_obj = []
        more_obj = []
        obj_pres = {0:[], 1:[], 2:[], 3:[]}
        obj_absent = {0:[], 1:[], 2:[], 3:[]}
        local_obj_count = {}
        for path in ann_paths:
            annotations_file = os.path.join(dir,magnification,path)
            annotations = np.loadtxt(annotations_file, ndmin=2, delimiter=" ")
            #if len(annotations) > 1:
            #    annotations = annotations.reshape(-1,5)
            #else:
            #    annotations = annotations.reshape(1,5)
                
            #get values for more images with more than 4 objects and images with less than 5 object
            if len(annotations) < 5:
                less_obj.append(annotations_file)
            else:
                more_obj.append(annotations_file)
                
            #get values for the class object counts at eaqch resolution
            present_classes = [-1,-1,-1,-1]
            for clas in annotations[:,0]:
                clas = int(clas)
                present_classes[clas] = clas
                if clas in local_obj_count.keys():
                    local_obj_count[clas]+=1
                else:
                    local_obj_count[clas]=1
                    
            #get value for files with no speciifc class and files with a speciifc class
            for i in range(len(present_classes)):
                if present_classes[i] == -1:
                    obj_absent[i].append(annotations_file)
                else:
                    obj_pres[i].append(annotations_file)
         
        #add the magnification wise details 
        object_count[magnification] = local_obj_count
        less_objects[magnification] = less_obj
        more_objects[magnification] = more_obj
        object_present[magnification] = obj_pres
        object_absent[magnification] = obj_absent
         
    return object_count, less_objects, more_objects, object_present, object_absent 


"""
the list in this hashmap will contain the
[total objects per class, the amount of more to be incremeted,
  total number times each object should be incremented,  
  total number times this particular object will be incremented to an image with less than 5 objects]
"""
def get_count_tobe_incremented(object_count, less_obj_images):
    increment_count = {'100x':{0:[], 1:[], 2:[], 3:[]}, 
                       '400x':{0:[], 1:[], 2:[], 3:[]}, '1000x':{0:[], 1:[], 2:[], 3:[]}}
    for magnification in object_count.keys():
        counts = object_count[magnification]
        maximum  = max(counts.values())
        for obj in counts.keys():
            if maximum > counts[obj]:
                num_less = maximum - counts[obj] #how many objects are less
                num_each_obj_incr = int(num_less / counts[obj]) # number of times each objects needs to be augmneted
                total_incr = num_each_obj_incr * counts[obj] # total count, might be less than maximum
                total_incr_in_each_img = math.ceil((counts[obj] * num_each_obj_incr)/len(less_obj_images[magnification]) )
                increment_count[magnification][obj] = [counts[obj],total_incr, num_each_obj_incr, total_incr_in_each_img]
                
    return increment_count
                       



def get_annotations_old(ann_file_path, image_shape):
    yolo_anns = np.loadtxt(ann_file_path, delimiter=" ")
    img_ann = np.array(yolo_anns.shape)
    image_width, image_height,_ = image_shape
    for i in range(len(yolo_anns)):
        x_center = yolo_anns[i][1]
        y_center = yolo_anns[i][2]
        width = yolo_anns[i][3]
        height = yolo_anns[i][4]
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        label_bbox = np.array([yolo_anns[i][0],x_min, y_min, x_max, y_max])
        img_ann[i] = label_bbox
    return img_ann




def get_annotations(ann_file_path, image_shape):
    yolo_anns = np.loadtxt(ann_file_path,ndmin=2, delimiter=" ")
    #print(yolo_anns)
    img_ann = np.ones((yolo_anns.shape))
    image_width, image_height,_ = image_shape
    #print(image_width, image_height)
    for i in range(len(yolo_anns)):
        x_center = yolo_anns[i][1]
        y_center = yolo_anns[i][2]
        width = yolo_anns[i][3]
        height = yolo_anns[i][4]
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        label_bbox = np.array([yolo_anns[i][0],x_min, y_min, x_max, y_max],dtype=int)
        img_ann[i] = label_bbox
    #print(img_ann.shape,img_ann)
    return img_ann





def paste_object(object_patch, object_ann,target_image, target_ann, target_img_empty_region, incr):
    rejion = target_img_empty_region[0]
    new_empty_rejions = [lst for lst in target_img_empty_region if lst != rejion]
    start_row = rejion[0] + 5
    start_col = 20
    x_min = start_row
    x_max = int(x_min + (object_ann[3]-object_ann[1])) 
    y_min = start_col
    y_max = int(y_min + (object_ann[4]-object_ann[2]))
    #print(x_min,x_max,y_min,y_max)
    #print(object_patch.shape)
    for i in range(incr):
        target_image[x_min:x_max , y_min:y_max] = object_patch
        new_ann = np.array([object_ann[0], x_min, y_min, x_max, y_max])
        target_ann = np.vstack([target_ann, new_ann])
        y_min = y_min + int(object_ann[4]-object_ann[2]) + 20
        y_max = y_min + int(object_ann[4]-object_ann[2])
        
    return target_image, target_ann, new_empty_rejions




def convert_to_yolo_file(size, ann, text_filename):
    data = []
    dw = 1./size[0]
    dh = 1./size[1]
    for i in range(len(ann)):
        x = (ann[i][1] + ann[i][3])/2.0
        y = (ann[i][2] + ann[i][4])/2.0
        w = ann[i][3] - ann[i][1]
        h = ann[i][4] - ann[i][2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        data.append(f"{int(ann[i][0])} {x} {y} {w} {h}\n")
    # Save the data to a JSON file
    with open(text_filename, "w") as text_file:
        text_file.writelines(data)


def copy_more_obj_data(more_obj_data):
    for magnification in more_obj_data.keys():
        files = more_obj_data[magnification]
        for file in files:
            label = file.replace("Transform_olympus", "Transform_olympus_aug")
            shutil.copyfile(file, label) #copy label
            sor_img = get_image(file)
            dest_img = sor_img.replace("Transform_olympus", "Transform_olympus_aug")
            shutil.copyfile(sor_img, dest_img) #copy image



def get_augemented_data(root_dir, dir):
    make_dirs(root_dir)
    #dict to record the files and corresponging object bounding box that has been incremented
    objects_incremented = {'1000x':{},'400x':{},'100x':{}}
    #get magnification wise preprocessed data
    object_count, objs_less_than_thres, objs_more_than_thres, object_present,_ = preprocess_data(dir)
    copy_more_obj_data(objs_more_than_thres)
    increment_stats = get_count_tobe_incremented(object_count,objs_less_than_thres)
    for magnificaton in objs_less_than_thres.keys():
        count = 0
        less_obj_label_paths = objs_less_than_thres[magnificaton] #labels of the images that have 4 objects only and now we are going to augment some more into it
        labs_with_clas_present = object_present[magnificaton] #[value for values in object_present[magnificaton].values() for value in values]
        #temp = set(labs_with_clas_present) # get unique ones
        #labs_with_clas_present = [elem for elem in temp] # conver to list

        for less_objs_file in less_obj_label_paths:
            path_img_with_less_obj = get_image(less_objs_file)
            less_objs_img = cv2.imread(path_img_with_less_obj)
            less_objs_anns = get_annotations(less_objs_file, less_objs_img.shape)
            
            empty_rejions = find_non_overlapping_position(less_objs_anns , less_objs_img.shape)
            if not empty_rejions: 
                count+=1
                #print("empty rejion again: ",len(empty_rejions), less_objs_anns.shape,less_objs_file )
                continue

            #get the image with label and its corresponding annottaion
            for clas in increment_stats[magnificaton].keys():
                list = increment_stats[magnificaton][clas]# list containg the total number of each class, how much more instances needed, how much each one has to be multiplied and how many will each augment in eah less obj image
                #print(list)
                if list: #if list not empty
                    incr_times = increment_stats[magnificaton][clas][-1] #num times this specific class has to be augmented
                    labs_with_clas = labs_with_clas_present[clas]
                    rand = random.randrange(len(labs_with_clas))
                    path_lab_with_obj = labs_with_clas[rand]
                    path_img_with_objs = get_image(path_lab_with_obj)
                    img_with_objs = cv2.imread(path_img_with_objs) 
                    ann_with_objs = get_annotations(path_lab_with_obj, img_with_objs.shape)
                    for ann in ann_with_objs:
                        count = 0 # if class 1 occurs twice, it should only be augemnted once in this particular image
                        if path_lab_with_obj in objects_incremented[magnificaton].keys():
                            l = objects_incremented[magnificaton][path_lab_with_obj]
                            if not any(np.array_equal(arr, ann) for arr in l):
                                #print('class of the object patch to incr: ',ann)
                                if int(ann[0]) == clas:
                                    #print(ann[1],ann[3],ann[2],ann[4])
                                    object_patch = img_with_objs[int(ann[1]):int(ann[3]),int(ann[2]):int(ann[4])]
                                    less_objs_img,less_objs_anns, empty_rejions = paste_object(object_patch,ann, less_objs_img,
                                                                                               less_objs_anns,empty_rejions, 
                                                                                                incr_times)
                                    objects_incremented[magnificaton][path_lab_with_obj].append(ann)
                                    break
                        else:
                            if int(ann[0]) == clas:
                                object_patch = img_with_objs[int(ann[1]):int(ann[3]),int(ann[2]):int(ann[4])]
                                #print(object_patch.shape)
                                less_objs_img,less_objs_anns, empty_rejions = paste_object(object_patch,ann, less_objs_img,
                                                                                        less_objs_anns,empty_rejions, 
                                                                                        incr_times)
                                objects_incremented[magnificaton][path_lab_with_obj] = [ann]
                                break
                        #delete the image with objs from the list img_with_clas_present
                        #if len(ann) == len(objects_incremented[magnificaton][path_lab_with_obj]):
                        #    object_present[magnificaton][clas].remove(path_lab_with_obj)
                            
            #save the new image and its annotation
            #delete the image with objs from the list img_with_clas_present
            #save the new image and its annotation
            new_lab_file = less_objs_file.replace("Transform_olympus", "Transform_olympus_aug")
            new_img_file = path_img_with_less_obj.replace("Transform_olympus", "Transform_olympus_aug")
        
            #convert annotations to yolo
            w,h,_ = less_objs_img.shape
            convert_to_yolo_file((w,h), less_objs_anns, new_lab_file)
            cv2.imwrite(new_img_file,less_objs_img)
    
        print(count)  

###########################################################################################      

if __name__ == '__main__':

    root_dir = Path("datasets/malaria_dataset")
    dir = Path("datasets/malaria_dataset/Transform_olympus/labels/train")
    get_augemented_data(root_dir,dir)