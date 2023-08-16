# EfficentDet Model Training
A tool that imports images, detects annotations on them using the coco efficientdet-lite4 model, and saves the annotations as XML files.

## Setup
1. Ensure Python 3.9 is installed: `sudo apt-get install python3.9 python3.9-distutils python3.9-dev`
2. Update alternatives: `sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1`
3. Install required Python packages: `pip install -r requirements.txt`
4. Install additional system packages: `sudo apt-get install python3-pip libportaudio2`

## Running the Training Script
1. Make sure your dataset paths are set correctly in `annotate.py`. This is when we make our directories and when set input_directory, output_images_directory, and output_xml_directory. 
2. Set your batch size and confidence threshold for boxes and classes
3. Run the training with: `python annotate.py`.
