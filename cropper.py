# use clt+shift+p to activate the conda environment
from traceback import print_exc
from factory import loader
from factory import cropper
import os
import argparse
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path

#{"LabelName": "Fruit","Subcategory": [{"LabelName": "Apple"},{"LabelName": "Grape"},{"LabelName": "Common fig"},{"LabelName": "Pear"},{"LabelName": "Strawberry"},{"LabelName": "Tomato"},{"LabelName": "Lemon"},{"LabelName": "Banana"},{"LabelName": "Orange"},{"LabelName": "Peach"},{"LabelName": "Mango"},{"LabelName": "Pineapple"},{"LabelName": "Grapefruit"},{"LabelName": "Pomegranate"},{"LabelName": "Watermelon"}
#def load(label_list, tag_list,max_samples,in_dir):

labels=["Apple","Grape","Pear","Strawberry","Tomato","Lemon","Banana","Orange","Peach","Mango"]
parser = argparse.ArgumentParser(description='Cropping OpenImages')
parser.add_argument("-i","--indir", type=str, help='Input dir for existing OpenImages directory', default= os.getcwd() )
parser.add_argument("-o","--outdir", type=str, help='Output dir for saving cropped image', default='/home/amir/Downloads/Fruit')
parser.add_argument("-m","--max_sample", type=int, help="Maximum number of samples",default = [10, 10, 10])
parser.add_argument("-t", "--loading_tags", type=list, help="Spliting loading data", default = ["train","validation"])
parser.add_argument("-l", "--loading_labels", type=list, help="List of loading labels", default = ["Banana"])
parser.add_argument("-s","--seed", type=int, default=51,  metavar="integer number")
args = parser.parse_args()



class Category:
    def __init__(self, tag, label):
        self.tag = tag
        self.label = label

    def make_folder(self, out_dir):
        self.saving_path = os.path.join(out_dir, self.tag, self.label)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
    
    def load(self, in_dir, seed, max_sample):
        self.view =  loader(self.label, self.tag, in_dir, seed, max_sample)

    def crop(self):
        for sample in self.view:
            sample = self.view.take(1).first()  
            path = sample.filepath 
            saving_name = sample.open_images_id
            i=0
            for detected in sample.detections.detections:
                if detected['label']== label:
                    boundingbox = detected['bounding_box']
                    cropper(saving_name+"_"+str(i)+".jpg",path,boundingbox,self.saving_path)
                    i=i+1           



for index,tag in enumerate(args.loading_tags):
    for label in args.loading_labels:
        slice = Category(tag, label)
        slice.load(args.indir, args.seed, args.max_sample[index])
        slice.make_folder(args.outdir)
        slice.crop()
        del slice

