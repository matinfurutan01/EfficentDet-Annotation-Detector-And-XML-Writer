# -*- coding: utf-8 -*-

# annotate.py
import os
import subprocess
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from xml.etree.ElementTree import Element, SubElement, ElementTree

model_url = "https://tfhub.dev/tensorflow/efficientdet/lite4/detection/2"
model = hub.load(model_url)

# Load the class names for Coco2017
class_names = {
    0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


# Define directories for processed images and XMLs
os.makedirs("/path/processed_images", exist_ok=True)
os.makedirs("/path/processed_images_xml", exist_ok=True)

#This function visualizes our detections
def visualize_detections(image, boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Cast to uint8 if needed
    image = tf.cast(image, tf.uint8)

    # Display the image without dividing by 255
    ax.imshow(image)

    # Loop through the detected bounding boxes
    for box in boxes:
        # Get box coordinates and draw a rectangle
        ymin, xmin, ymax, xmax = box
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()



# Define filenames_batch to store the filenames
def get_batched_images(input_directory, batch_size):
    images_batch = []
    filenames_batch = [] # Store filenames in this batch
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg"):
            # Load image
            image_path = os.path.join(input_directory, filename)
            image = load_img(image_path, target_size=(1080, 1920)) # Load with original size
            image = img_to_array(image).astype(np.uint8) # Convert to uint8

            # Add to batch without preprocessing
            images_batch.append(image)
            filenames_batch.append(filename) # Append filename to batch

            # Yield batch when it's full
            if len(images_batch) == batch_size:
                yield np.array(images_batch), filenames_batch
                images_batch = []
                filenames_batch = []
    # Yield any remaining images
    if images_batch:
        yield np.array(images_batch), filenames_batch


# Directory containing the images
input_directory = '/path/images'
output_images_directory = '/path/processed_images'
output_xml_directory = '/path/processed_images_xml'
os.makedirs(output_images_directory, exist_ok=True)
os.makedirs(output_xml_directory, exist_ok=True)

batch_size = 8  # batch size of images our model is creating annotations/xmls for -- the larger it is the more memory it will use

for images_batch, filenames_batch in get_batched_images(input_directory, batch_size):
    # Convert the batch of images to a tensor
    images_batch_tensor = tf.stack(images_batch)

    # Run detections on the batch
    boxes, scores, classes, num_detections = model(images_batch_tensor)


    for filename, image, box, score, cls in zip(filenames_batch, images_batch, boxes.numpy(), scores.numpy(), classes.numpy()):
        # Filter boxes/classes with confidence above a set threshold
        filtered_boxes = [b for b, s in zip(box, score) if s > 0.3]
        filtered_classes = [int(c) for c, s in zip(cls, score) if s > 0.3]

        visualize_detections(image, filtered_boxes)

        # Create XML
        annotation = Element("annotation")

        # Add size element
        size = SubElement(annotation, "size")
        SubElement(size, "width").text = str(1920) # Original width
        SubElement(size, "height").text = str(1080) # Original height
        SubElement(size, "depth").text = str(3)

        for i, (box, class_idx) in enumerate(zip(filtered_boxes, filtered_classes)):
            object = SubElement(annotation, "object")
            # Use the class index to get the label name
            SubElement(object, "name").text = class_names[class_idx]
            bndbox = SubElement(object, "bndbox")
            # Use the round function to round the coordinates to integers
            SubElement(bndbox, "xmin").text = str(round(box[1]))
            SubElement(bndbox, "ymin").text = str(round(box[0]))
            SubElement(bndbox, "xmax").text = str(round(box[3]))
            SubElement(bndbox, "ymax").text = str(round(box[2]))

        # Save XML
        xml_filename = os.path.join(output_xml_directory, os.path.splitext(filename)[0] + '.xml') # use output_xml_directory
        tree = ElementTree(annotation)
        tree.write(xml_filename)







    # save the processed images
    for filename, processed_image in zip(filenames_batch, images_batch):
        if filename.endswith(".jpg"):
            save_path = os.path.join(output_images_directory, filename) # use output_images_directory
            save_img(save_path, processed_image)

