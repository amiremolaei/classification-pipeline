# use clt+shift+p to activate the conda environment
from cProfile import label
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import patches
from PIL import Image
import os


def cropper(name,path,box):
    im = Image.open(path)
    width, height = im.size
    x = int(box[0]*width)
    w = int(box[2]*width)
    y = int(box[1]*height)
    h = int(box[3]*height)
    area = (x, y, x+w, y+h)
    cropped_img = im.crop(area)
    cropped_img.save(name)


directory = os.getcwd()
fo.config.dataset_zoo_dir = directory # specifying the directory for saving downloaded data

# if the images are to be downloaded
dataset = foz.load_zoo_dataset(
   "open-images-v6",
    split="validation",
    label_types=["detections"],
    classes = ["Burrito"],
    max_samples=100,
    seed=51,
    shuffle=True,
    dataset_name="open-images-food",
)

# if the images are loaded from a local drive (not yet completed)

#dataset = fo.Dataset.from_dir(
#    dataset_dir= os.path.join (directory,"open-images-v6","validation"),  dataset_type=fo.types.FiftyOneImageClassificationDataset)

view = dataset.view()

for sample in view:
    sample = view.take(1).first()
    #print ((sample.id))
    print(sample.open_images_id)
    path = sample.filepath
    #print(type(sample.detections))
    
    saving_name = sample.open_images_id
    i=0
    for detected in sample.detections.detections:
        if detected['label']=='Burrito':
            boundingbox = detected['bounding_box']
            cropper(saving_name+"_"+str(i)+".jpg",path,boundingbox)
            i=i+1

# if we want to visualize the loaded data within the app the following uncomment following

#if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
#    session = fo.launch_app(dataset)
#   session.wait()