import os
import shutil
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET
############################################################################################

def xyxy_to_yolo(x1, y1, x2, y2, img_width, img_height):
    # Calculate YOLO format values
    center_x = (x1 + x2) / 2 / img_width
    center_y = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return center_x, center_y, width, height

def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)
   
   
def make_xml(dataxyxy, output_dir): #img_file_name} {crop_width} {crop_height} {cat_name} {crop_bbox[0]} {crop_bbox[1]} {crop_bbox[2]} {crop_bbox[3]
    dataset = {
        'image_filename': "",
        'image_width': 0,
        'image_height': 0,
        'objects':[]
    }
    for l in dataxyxy:
        dataset['image_filename'] = l[0]
        dataset['image_width'] = l[1]
        dataset['image_height'] = l[2]
        dataset['objects'].append({'name':l[3],'xmin': l[4], 'ymin': l[5], 'xmax': l[6], 'ymax':l[7] })
        
    
    image_filename = dataset['image_filename']
    image_width = dataset['image_width']
    image_height = dataset['image_height']
    objects = dataset['objects']

    xml_tree = create_xml_annotation(image_filename, image_width, image_height, objects)
    annotation_filename = os.path.splitext(os.path.basename(image_filename))[0] + '.xml'
    save_path = output_dir.replace(".json",".xml")
    xml_tree.write(save_path)
    #xml_tree.write(os.path.join(output_dir, annotation_filename))
     
        
def create_xml_annotation(image_filename, image_width, image_height, objects):
    annotation = ET.Element('annotation')
    # Add basic image information
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.dirname(image_filename)
    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_filename)
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_width)
    height = ET.SubElement(size, 'height')
    height.text = str(image_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'  # Assuming RGB images
    # Add object annotations
    for obj in objects:
        obj_elem = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj_elem, 'name')
        name.text = obj['name']
        bndbox = ET.SubElement(obj_elem, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(obj['xmin']))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(obj['ymin']))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(obj['xmax']))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(obj['ymax']))
    # Create XML tree
    tree = ET.ElementTree(annotation)
    return tree

        
 
def modify_json(input_json_file, output_json_file, data_xyxy):
    with open(output_json_file, 'w') as f:
        json.dump(data_xyxy, f, indent=4)
        
 
def transform_bbox(original_bbox, crop_center, crop_size, second_microscope=False):
    """
    Transform bounding box coordinates from the original image to the cropped patch.

    Args:
    - original_bbox (tuple): Bounding box coordinates (x1, y1, x2, y2) in the original image.
    - crop_center (tuple): Coordinates of the center of the cropped patch (center_x, center_y).
    - crop_size (tuple): Size of the cropped patch (crop_width, crop_height).

    Returns:
    - tuple: Transformed bounding box coordinates in the cropped patch (x1_cropped, y1_cropped, x2_cropped, y2_cropped).
    """
    x1, y1, x2, y2 = original_bbox
    center_x, center_y = crop_center
    if second_microscope:
         crop_height, crop_width, _ = crop_size
    else:
        crop_width, crop_height = crop_size
    # Calculate the shift in origin caused by center cropping
    shift_x = center_x - crop_width / 2
    shift_y = center_y - crop_height / 2
    # Transform bounding box coordinates
    x1_cropped = max(1, x1 - shift_x)
    y1_cropped = max(1, y1 - shift_y)
    x2_cropped = min(crop_width, x2 - shift_x)
    y2_cropped = min(crop_height, y2 - shift_y)
    return x1_cropped, y1_cropped, x2_cropped, y2_cropped       
        
        

def crop_center(orig_img, second_microscope=False, w_ratio=3/6, h_ratio=4/5):
    if second_microscope:
        height, width,_ = orig_img.shape
    else:
        width, height = orig_img.size
    # Calculate the coordinates for the rectangular patch
    w_center = int(width/2)
    h_center = int(height/2)
    if second_microscope:
        new_w = int(width * h_ratio)
        new_h = int(height * w_ratio)
    else:
        new_w = int(width * w_ratio)
        new_h = int(height * h_ratio)
        
    x_start =  w_center - int(new_w/2)
    x_end = w_center + int(new_w/2) 
    y_start =  h_center - int(new_h/2)
    y_end = h_center + int(new_h/2) 
    # Extract the rectangular patch
    if second_microscope:
        cropped_image = orig_img[y_start:y_end, x_start:x_end]
        height_c, width_c,_ = cropped_image.shape
    else:
        cropped_image = orig_img.crop((x_start,y_start,x_end, y_end ))
        width_c, height_c = cropped_image.size
    return cropped_image, w_center, h_center, width_c, height_c
    


def get_crop_and_bbox(json_label_path,label_map, save_path, orig_image, save_img_path, second_microscope=False):
    try:
        with open(json_label_path, 'r') as json_file:
            ann = json.load(json_file)
    except:
        #print("failed to read: ", xml_label_path)
        return 0
    crop_done = False
    cat_id = -1
    data = []
    data_xyxy = []
    for ki in ann.keys():
        if "Cell_" in ki:
            for k,v in ann[ki].items():
                if k == "Label2" and v in label_map.keys():
                    cat_id = label_map[v]
                    cat_name = v
                    if not crop_done:
                        croped_img, orig_w_cent, orig_h_cent, crop_width, crop_height = crop_center(orig_image,second_microscope)
                        crop_done = True
                elif k == "x1":
                    x1 = v
                elif k == "x2":
                    x2 = v
                elif k == "y1":
                    y1 = v
                elif k == "y2":
                    y2 = v
            if cat_id != -1 and crop_done:
                orig_box = [int(x1),int(y1),int(x2),int(y2)]
                if second_microscope:
                    crop_bbox = transform_bbox(orig_box,(orig_w_cent, orig_h_cent) , croped_img.shape, second_microscope)
                    x,y,w,h = xyxy_to_yolo(*crop_bbox, croped_img.shape[1] , croped_img.shape[0])
                else:
                    crop_bbox = transform_bbox(orig_box,(orig_w_cent, orig_h_cent) , croped_img.size)
                    x,y,w,h = xyxy_to_yolo(*crop_bbox, croped_img.size[0] , croped_img.size[1])
                
                
                
                data.append(f"{cat_id} {x} {y} {w} {h}\n")
                img_file_name = json_label_path.replace(".json",".jpg")
                data_xyxy.append([img_file_name, crop_width, crop_height, cat_name,crop_bbox[0],crop_bbox[1], crop_bbox[2], crop_bbox[3]])
                cat_id = -1
            
    # Save the data to a JSON file
    if data and crop_done:
        if second_microscope:
            cv2.imwrite(save_img_path, croped_img)
        else:
            croped_img.save(save_img_path)
        save_path_yolo = save_path.replace(".json",".txt")
        with open(save_path_yolo, "w") as text_file:
            text_file.writelines(data)
        json_save_path = save_path.replace("labels", "l_jsons")
        xml_save_path = save_path.replace("labels", "l_xmls")
        modify_json(json_label_path, json_save_path, data_xyxy)
        make_xml(data_xyxy, xml_save_path)
        #shutil.copyfile(json_label_path, json_save_path)
        return 1
    return 0
        
#############################################################################

def make_dirs(parent):
    #image dirs for china at each resolution and chucnk
    mkdir(parent/'rabin_First'/'images')
    mkdir(parent/'rabin_Second'/'images')
    mkdir(parent/'rabin_First'/'labels')
    mkdir(parent/'rabin_Second'/'labels')
    mkdir(parent/'rabin_First'/'l_jsons')
    mkdir(parent/'rabin_Second'/'l_jsons')
    mkdir(parent/'rabin_First'/'l_xmls')
    mkdir(parent/'rabin_Second'/'l_xmls')
    
    
"""prerpocess the source data
   """
def preprocess_full_data(parent,dest_dir,label_map):
    
    make_dirs(dest_dir)

    for dir in os.listdir(parent): # first, second
        count = 0
        image_list_all= []
        for data_dirs in os.listdir(parent / dir): # sub dirs containing images and jsons
            imgs_path = sorted(os.listdir(parent / dir / data_dirs / "images" ))
            labels_path = sorted(os.listdir(parent / dir / data_dirs / "jsons" )) # root_dir / getfine / test / berlin
            image_list = []
            for img,l in zip(imgs_path ,labels_path ):
                if img.split('.')[0] == l.split('.')[0]:
                    label =  os.path.join(parent / dir / data_dirs / "jsons" / l)
                    img_path = os.path.join(parent / dir / data_dirs / "images" / img)
                    
                    if dir == "first":
                        im = Image.open(img_path)
                        save_dir = os.path.join(dest_dir / 'rabin_First')
                        save_lab_path = os.path.join(dest_dir / 'rabin_First' / 'labels' / l)
                        save_img_path = os.path.join(dest_dir / 'rabin_First' / 'images'/ img )
                        flag = get_crop_and_bbox(label,label_map, save_lab_path, im, save_img_path)
                    elif dir == "second":
                        im = cv2.imread(img_path)
                        save_dir = os.path.join(dest_dir / 'rabin_Second')
                        save_lab_path = os.path.join(dest_dir / 'rabin_Second' / 'labels' / l )
                        save_img_path = os.path.join(dest_dir / 'rabin_Second' / 'images'/ img )
                        flag = get_crop_and_bbox(label,label_map, save_lab_path, im, save_img_path, second_microscope=True)
                    if flag:
                        #shutil.copyfile(img_path, save_img_path)
                        image_list.append(f'images/{img}\n')
                    else:
                        count+=1
                
            image_list_all.extend(image_list)        
        with open(save_dir+'/yolo.txt', 'w') as f:  
            f.writelines(image_list_all)
        print(count, " json files could not be read in ", dir)    
    
        with open(save_dir+'/classes.txt', 'w') as f:
            for k, v in label_map.items():
                f.write(f'{k}\n')
    print([k for k in label_map.keys()], len([k for k in label_map.keys()]))    


if __name__ == '__main__':
    #root_dir = Path(__file__).parent
    """
    Instruction:
    1: Download the original cityscapes dataset like the following structure:
    -- cityscapes
       -- gtFine
       -- leftImg8bit
       -- vehicle_trainextra
       -- ...
    2: Change the root_dir to your dataset's path
    3: python cityscapes_to_yolo.py
    """
    
    label_map = {'Large Lymph':0, 'Neutrophil':1, 'Small Lymph':2, 'Monocyte':3}

    root_dir = Path('/media/waqas/Jav/Sumayya/raabin_jsons')
    dest_dir = Path("/media/waqas/Jav/Sumayya/blood_cancer/YOLOv5_aug/datasets")
    preprocess_full_data(root_dir,dest_dir, label_map)