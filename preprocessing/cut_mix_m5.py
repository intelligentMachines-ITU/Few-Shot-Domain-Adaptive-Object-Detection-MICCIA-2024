#apply cut-paste augmnetion on only blurred source data or real source data
#cut some context also from the patch
#apply patch ramdomly not just in the above sections

import cv2
import os
import shutil
import random
import numpy as np
import math
from pathlib import Path
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
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
    mkdir(parent/'Olympus_tar_aug'/'images'/'train'/'100x')
    mkdir(parent/'Olympus_tar_aug'/'images'/'train'/'400x')
    mkdir(parent/'Olympus_tar_aug'/'images'/'train'/'1000x')
    
    #label dirs for olympus at each resolution and chucnk
    mkdir(parent/'Olympus_tar_aug'/'labels'/'train'/'100x')
    mkdir(parent/'Olympus_tar_aug'/'labels'/'train'/'400x')
    mkdir(parent/'Olympus_tar_aug'/'labels'/'train'/'1000x')


def preprocess_source_data(dir):
    object_count = {}
    less_objects = {}
    more_objects = {}
    object_present = {}
    object_absent = {} 
    #get data for further preprocessing
    for magnification in os.listdir(dir): 
        if magnification == '1000x.cache':
            continue
        ann_paths = os.listdir(os.path.join(dir,magnification))
        less_obj = []
        more_obj = []
        obj_pres = {0:[], 1:[], 2:[], 3:[]}
        obj_absent = {0:[], 1:[], 2:[], 3:[]}
        local_obj_count = {}
        for path in ann_paths:
            annotations_file = os.path.join(dir,magnification,path)
            annotations = np.loadtxt(annotations_file, ndmin=2, delimiter=" ")
            #get values for more images with more than 4 objects and images with less than 5 object
            if len(annotations) < 5:
                less_obj.append(annotations_file)
            else:
                more_obj.append(annotations_file)
            #get values for the class object counts at each resolution
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



def get_target_few_data(target_lab_dir):
    few_labs_all_res = {'100x':[],'400x':[], '1000x':[]}
    #get data for further preprocessing
    for magnification in os.listdir(target_lab_dir): 
        ann_paths = os.listdir(os.path.join(target_lab_dir,magnification))
        for path in ann_paths:
            annotations_file = os.path.join(target_lab_dir,magnification,path)
            few_labs_all_res[magnification].append(annotations_file)  
    return few_labs_all_res


def copy_more_obj_data(more_obj_data):
    for magnification in more_obj_data.keys():
        files = more_obj_data[magnification]
        for file in files:
            label = file.replace("HCM_1000_source_only", "Olympus_tar_aug")
            shutil.copyfile(file, label) #copy label
            sor_img = get_image(file)
            dest_img = sor_img.replace("HCM_1000_source_only", "Olympus_tar_aug")
            shutil.copyfile(sor_img, dest_img) #copy image



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
                total_incr_in_each_img = int(total_incr / len(less_obj_images[magnification])) if int(total_incr / len(less_obj_images[magnification])) > 0 else 1
                times_each_obj_incr_in_img = int(num_each_obj_incr / total_incr_in_each_img)
                increment_count[magnification][obj] = [counts[obj], times_each_obj_incr_in_img, total_incr_in_each_img]
                
    return increment_count



def get_annotations(ann_file_path, image_shape):
    yolo_anns = np.loadtxt(ann_file_path,ndmin=2, delimiter=" ")
    #print("yolo format anns: ",yolo_anns)
    img_ann = np.ones((yolo_anns.shape))
    image_width,image_height = image_shape
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


def find_empty_positions(dir,bboxes, image_shape):
    # Find non-overlapping position in the background
    mask = np.ones((image_shape[1],image_shape[0]))
    for box_coord in bboxes:
        mask[int(box_coord[2]) :int(box_coord[4]) ,int(box_coord[1]) :int(box_coord[3])] = 0
    #cv2.imwrite(dir+'/empty_mask.png',mask*255)
    return mask


def paste_object(coords, object_patch, object_ann,image,ann):
    x_min,y_min,x_max,y_max = coords 
    image.paste(object_patch, (x_min,y_min))
    new_ann = np.array([object_ann[0], x_min, y_min, x_max, y_max])
    ann = np.vstack([ann, new_ann])
    return image, ann



def convert_to_yolo_file(size, ann, text_filename): #ann = [object_ann[0], x_min, y_min, x_max, y_max]
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




def find_pasting_rejion(dir, mask, patch_shape):
    w,h = patch_shape # rowxcolx3
    found = False
    temp = np.ones((h,w))
    for _ in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            rand_row = random.randrange(mask.shape[0])
            rand_col = random.randrange(mask.shape[1])
            empty_patch = mask[rand_row:rand_row+h, rand_col:rand_col+w]
            if empty_patch.shape == temp.shape and np.all(empty_patch == 1):
                mask[rand_row:rand_row+h, rand_col:rand_col+w] = 0
                #cv2.imwrite(dir+'/mask_appended.png',mask*255)
                x_min,y_min,x_max,y_max = rand_col,rand_row, rand_col+w, rand_row+h 
                found =  True
                return mask, [x_min,y_min,x_max,y_max], found
    return False,False, found


def random_color_and_blur_variation(image_patch, intensity_range=(0.3, 1.7), blur_range=(1, 4), tar=False, upsample=False):
    # Apply random color variation
    enhancer = ImageEnhance.Color(image_patch)
    color_factor = np.random.uniform(*intensity_range)
    patch = enhancer.enhance(color_factor)
    if not tar:
        # Apply random blurring effect
        blur_radius = np.random.uniform(*blur_range)
        patch = patch.filter(ImageFilter.GaussianBlur(blur_radius))

    if upsample:
        upsampling_factor = 2  # Adjust as needed
        # Upsample the patch using Lanczos interpolation
        patch = patch.resize((patch.width * upsampling_factor,
                              patch.height * upsampling_factor), Image.BICUBIC)

    return patch



def random_color_variation(image_patch, intensity_range=(0.3, 1.7), tar=False):
    upsampling_factor = 2  # Adjust as needed
    enhancer = ImageEnhance.Color(image_patch)
    factor = np.random.uniform(*intensity_range)
    patch = enhancer.enhance(factor)
    if tar:
        # Upsample the patch using Lanczos interpolation
        patch = patch.resize((patch.width * upsampling_factor
                                    ,patch.height * upsampling_factor),Image.BICUBIC)
    return patch


def blur_img(mag,img):
    if mag == '1000x':
        img = img.filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR)
    elif mag == '400x':
        img = img.filter(ImageFilter.BLUR).filter(ImageFilter.BLUR)
    elif mag == '100x':
        img = img.filter(ImageFilter.BLUR)
    return img
    


def get_augemented_data(root_dir, sor_lab_dir, tar_lab_dir,dir, blur=False):
    make_dirs(root_dir)
    #dict to record the files and corresponging object bounding box that has been incremented
    objects_incremented = {'1000x':{},'400x':{},'100x':{}}
    #get magnification wise preprocessed data
    object_count_sor, objs_less_than_thres, objs_more_than_thres, object_present,_ = preprocess_source_data(sor_lab_dir)
    few_labs_all_res = get_target_few_data(tar_lab_dir)
    copy_more_obj_data(objs_more_than_thres)
    increment_stats = get_count_tobe_incremented(object_count_sor,objs_less_than_thres) # stats contain [original-count, number of image files a particular object has to be incremented, number of times one object has to be increemetd in an image]
    
    for magnificaton in objs_less_than_thres.keys():
        less_obj_label_paths = objs_less_than_thres[magnificaton] #labels of the images that have 4 objects only and now we are going to augment some more into it
        labs_with_clas_present = object_present[magnificaton] #[value for values in object_present[magnificaton].values() for value in value
        for less_objs_file in less_obj_label_paths:
            path_img_with_less_obj = get_image(less_objs_file)
            less_objs_img =  Image.open(path_img_with_less_obj)
            if blur:
                less_objs_img = blur_img(magnificaton,less_objs_img)
            less_objs_anns = get_annotations(less_objs_file, less_objs_img.size) #[label,x_min, y_min, x_max, y_max]
            empty_rejion_mask = find_empty_positions(dir, less_objs_anns , less_objs_img.size)
            
            #get the image with label and its corresponding annottaion
            for clas in increment_stats[magnificaton].keys():
                list = increment_stats[magnificaton][clas]# list containg the total number of each class, how much more instances needed, how much each one has to be multiplied and how many will each augment in eah less obj image
                #print(list)
                if list: #if list not empty
                    incr_times = increment_stats[magnificaton][clas][-1] #num times this specific class has to be augmented
                    incr_files = increment_stats[magnificaton][clas][1]
                    labs_with_clas = labs_with_clas_present[clas]
                    rand = random.randrange(len(labs_with_clas))
                    path_lab_with_obj = labs_with_clas[rand]
                    path_img_with_objs = get_image(path_lab_with_obj)
                    img_with_objs = Image.open(path_img_with_objs) 
                    if blur:
                       img_with_objs = blur_img(magnificaton,img_with_objs)
                    objs_ann = get_annotations(path_lab_with_obj, img_with_objs.size)#[label,x_min,y_min,x_max,y_max]
                    
                    for i in range(len(objs_ann)):
                        if int(objs_ann[i][0]) == clas:
                            ann = objs_ann[i]
                            if path_lab_with_obj in objects_incremented[magnificaton].keys():
                                dict = objects_incremented[magnificaton][path_lab_with_obj]
                                if clas not in dict.keys():
                                    ymin,ymax,xmin,xmax = int(ann[2]), int(ann[4]) , int(ann[1]), int(ann[3])
                                    object_patch = img_with_objs.crop((xmin, ymin, xmax, ymax))
                                    #object_patch = random_color_variation(object_patch)
                                    object_patch = random_color_and_blur_variation(object_patch)
                                    for _ in range(incr_times):
                                        empty_rejion_mask, cords, F = find_pasting_rejion(dir,empty_rejion_mask, object_patch.size) #x_min,y_min,x_max,y_max
                                        if F:
                                            less_objs_img,less_objs_anns = paste_object(cords, object_patch, ann,less_objs_img,less_objs_anns)
        
                                    objects_incremented[magnificaton][path_lab_with_obj][clas] = {i:1}
                                    break
                                elif i not in dict[clas].keys():
                                    ymin,ymax,xmin,xmax = int(ann[2]), int(ann[4]) , int(ann[1]), int(ann[3])
                                    object_patch = img_with_objs.crop((xmin, ymin, xmax, ymax))
                                    #object_patch = random_color_variation(object_patch)
                                    object_patch = random_color_and_blur_variation(object_patch)
                                    for _ in range(incr_times):
                                        empty_rejion_mask, cords, F = find_pasting_rejion(dir,empty_rejion_mask, object_patch.size) #x_min,y_min,x_max,y_max
                                        if F:
                                            less_objs_img,less_objs_anns = paste_object(cords, object_patch, ann,less_objs_img,less_objs_anns)
        
                                    objects_incremented[magnificaton][path_lab_with_obj][clas][i] = 1
                                    break
                                elif i in dict[clas].keys():
                                    if objects_incremented[magnificaton][path_lab_with_obj][clas][i] < incr_files:
                                        ymin,ymax,xmin,xmax = int(ann[2]), int(ann[4]) , int(ann[1]), int(ann[3])
                                        object_patch = img_with_objs.crop((xmin, ymin, xmax, ymax))
                                        #object_patch = random_color_variation(object_patch)
                                        object_patch = random_color_and_blur_variation(object_patch)
                                        for _ in range(incr_times):
                                            empty_rejion_mask, cords, F = find_pasting_rejion(dir,empty_rejion_mask, object_patch.size) #x_min,y_min,x_max,y_max
                                            if F:
                                               less_objs_img,less_objs_anns = paste_object(cords, object_patch, ann,less_objs_img,less_objs_anns)
        
                                        objects_incremented[magnificaton][path_lab_with_obj][clas][i]+=1
                                        break
                             
                            else:
                                    ymin,ymax,xmin,xmax = int(ann[2]), int(ann[4]) , int(ann[1]), int(ann[3])
                                    object_patch = img_with_objs.crop((xmin, ymin, xmax, ymax))
                                    #object_patch = random_color_variation(object_patch)
                                    object_patch = random_color_and_blur_variation(object_patch)
                                    for _ in range(incr_times):
                                        empty_rejion_mask, cords, F = find_pasting_rejion(dir,empty_rejion_mask, object_patch.size) #x_min,y_min,x_max,y_max
                                        if F:
                                            less_objs_img,less_objs_anns = paste_object(cords, object_patch, ann,less_objs_img,less_objs_anns)
        
                                    objects_incremented[magnificaton][path_lab_with_obj] = {}
                                    objects_incremented[magnificaton][path_lab_with_obj][clas]={i:1}
                                    
                                    break
                        
            #save the new image and its annotation
            #delete the image with objs from the list img_with_clas_present
            #save the new image and its annotation
            new_lab_file = less_objs_file.replace("HCM_1000_source_only", "Olympus_tar_aug")
            new_img_file = path_img_with_less_obj.replace("HCM_1000_source_only", "Olympus_tar_aug")
            
            #add patches from real few target sample at the specicifc resolution
            real_tar = few_labs_all_res[magnificaton]
            rand = random.randrange(len(real_tar))
            path_lab_real_tar = real_tar[rand]
            path_img_tar = get_image(path_lab_real_tar)
            tar_img = Image.open(path_img_tar)
            tar_ann = get_annotations(path_lab_real_tar, tar_img.size)#[label,x_min,y_min,x_max,y_max]
            if len(tar_ann)>0:
                rand_ann = random.randrange(len(tar_ann))
                t_ann = tar_ann[rand_ann]
                ymin,ymax = int(t_ann[2]), int(t_ann[4])
                xmin,xmax = int(t_ann[1]), int(t_ann[3])
                tar_patch = tar_img.crop((xmin, ymin, xmax, ymax))
                if tar_patch.size[0] > 0 and tar_patch.size[1] > 0:
                    #tar_patch = random_color_variation(tar_patch, tar=False) # set tar to true if you wan to double the size
                    tar_patch = random_color_and_blur_variation(tar_patch, tar=True)
                    for _ in range(1):
                        empty_rejion_mask, cords, F = find_pasting_rejion(dir,empty_rejion_mask, tar_patch.size) #x_min,y_min,x_max,y_max
                        if F:
                            less_objs_img,less_objs_anns = paste_object(cords, tar_patch, t_ann,less_objs_img,less_objs_anns)
            
                                                           
            #convert annotations to yolo
            w,h = less_objs_img.size
            convert_to_yolo_file((w,h), less_objs_anns, new_lab_file)
            new_img_file,less_objs_img.save(new_img_file)
            #cv2.imwrite(new_img_file,less_objs_img) 

###########################################################################################      

if __name__ == '__main__':
    dir = 'data_visualization'
    root_dir = Path('datasets')
    sor_dir = Path('datasets/HCM_1000_source_only/labels/train')
    tar_dir = Path('datasets/random_2_shot/labels/train')
    blur = False
    get_augemented_data(root_dir,sor_dir,tar_dir, dir,blur)  