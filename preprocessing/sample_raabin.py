import os
import shutil
import random

def copy_random_images(source_image_folder, source_label_folder,source_json_folder,source_xml_folder, dest_folder, num_images_to_copy):
    # Get the list of all images in the source folder
    image_files = os.listdir(source_image_folder)

    # Randomly select 100 images
    selected_images = random.sample(image_files, num_images_to_copy)

    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder+"/images", exist_ok=True)
    os.makedirs(dest_folder+"/labels", exist_ok=True)
    os.makedirs(dest_folder+"/l_jsons", exist_ok=True)
    os.makedirs(dest_folder+"/l_xmls", exist_ok=True)

    # Lists to store paths for creating val.txt
    image_paths = []
    
    # Copy selected images and labels to the destination folder
    for image_file in selected_images:
        # Copy image
        source_image_path = os.path.join(source_image_folder, image_file)
        dest_image_path = os.path.join(dest_folder, 'images', image_file)
        shutil.copy(source_image_path, dest_image_path)

        # Corresponding label file (assuming label files have the same name as images with a different extension)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        source_label_path = os.path.join(source_label_folder, label_file)
        dest_label_path = os.path.join(dest_folder, 'labels', label_file)
        shutil.copy(source_label_path, dest_label_path)
        
        # Corresponding label file (assuming label files have the same name as images with a different extension)
        label_file = os.path.splitext(image_file)[0] + '.json'
        source_json_path = os.path.join(source_json_folder, label_file)
        dest_label_path = os.path.join(dest_folder, 'l_jsons', label_file)
        shutil.copy(source_json_path, dest_label_path)
        
        # Corresponding label file (assuming label files have the same name as images with a different extension)
        label_file = os.path.splitext(image_file)[0] + '.xml'
        source_xml_path = os.path.join(source_xml_folder, label_file)
        dest_label_path = os.path.join(dest_folder, 'l_xmls', label_file)
        shutil.copy(source_xml_path, dest_label_path)

        # Store image paths for creating val.txt
        image_paths.append(f'images/{image_file}')
        
        # Delete the original image and label
        os.remove(source_image_path)
        os.remove(source_label_path)
        os.remove(source_json_path)
        os.remove(source_xml_path)

    # Write image paths to val.txt
    val_txt_path = os.path.join(dest_folder, 'test.txt')
    with open(val_txt_path, 'w') as val_txt_file:
        val_txt_file.write('\n'.join(image_paths))

    print(f"{num_images_to_copy} images and labels copied to {dest_folder}")
    print(f"val.txt created in {dest_folder}")

def main():
    
    source_image_folder = 'datasets/raabin/target/images'
    source_label_folder = 'datasets/raabin/target/labels'
    source_json_folder = 'datasets/raabin/target/l_jsons'
    source_xml_folder = 'datasets/raabin/target/l_xmls'
    dest_folder = 'datasets/raabin/target_test'
    num_images_to_copy = 450

    copy_random_images(source_image_folder, source_label_folder,source_json_folder,source_xml_folder, dest_folder, num_images_to_copy)

if __name__ == "__main__":
    main()
