# use clt+shift+p to activate the conda environment
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
import os
from fiftyone import ViewField as F



def loader(label, tag, indir, seed, max_samples):
    fo.config.dataset_zoo_dir = indir 
    dataset = foz.load_zoo_dataset(
    "open-images-v6",
        splits= tag,
        label_types=["detections"],
        classes = label,
        seed= seed,
        max_samples= max_samples,
        shuffle= True,
) 
    return dataset.view()


def cropper(name,path,box,saving_path):
    im = Image.open(path)
    width, height = im.size
    x = int(box[0]*width)
    w = int(box[2]*width)
    y = int(box[1]*height)
    h = int(box[3]*height)
    area = (x, y, x+w, y+h)
    cropped_img = im.crop(area)
    os.chdir(saving_path)
    cropped_img.save(name)
