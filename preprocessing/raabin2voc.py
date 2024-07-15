import os
import xml.etree.ElementTree as ET

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
        xmin.text = str(obj['xmin'])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(obj['ymin'])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(obj['xmax'])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(obj['ymax'])

    # Create XML tree
    tree = ET.ElementTree(annotation)
    return tree

# Example usage
image_filename = 'example.jpg'
image_width = 640
image_height = 480
objects = [
    {'name': 'Malaria_tar', 'xmin': 100, 'ymin': 200, 'xmax': 300, 'ymax': 400},
    {'name': 'Malaria_tar', 'xmin': 50, 'ymin': 100, 'xmax': 200, 'ymax': 250}
]

xml_tree = create_xml_annotation(image_filename, image_width, image_height, objects)
xml_tree.write('annotation.xml')
