import os
import glob
import numpy as np
from skimage import io,transform

class img_class:
    data = None
    label = None
    path = None

    def __init__(self,data,label,path):
        self.data = data
        self.label = label
        self.path = path

# Read imgs from test-classes folders as an array
# Possible Problem: test_classes: NOT foler's name, but folder name's alphabetic order
def read_img(path, max_img_per_folder = 99999, print_file_read = True, test_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
             width = 224, height = 224):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    paths = []
    cate = np.sort(cate)

    #The root category
    for idx, folder in enumerate(cate):
        i = 1
        # Label all the files in the sub folder
        for root, subFolers, files in os.walk(folder):
            for file in files:
                if (os.path.splitext(file)[1] in ['.jpg', '.bmp', '.png', 'jpeg']):
                    im = os.path.join(root,file)
                    if (idx in test_classes):
                        if (print_file_read):
                            print('Reading image %d:%s' % (i, im))
                        img = io.imread(im)
                        img = transform.resize(img, [width, height])
                        imgs.append(img)
                        paths.append(im)
                        labels.append(idx)
                        i += 1
                        if (i > max_img_per_folder):
                            print("Stop read img from folder %s" % folder)
                            break

    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32), np.asarray(paths, np.str), len(cate)

def read_one_img(path,width=224,height=224):
    img = io.imread(path)
    img = transform.resize(img, [width, height])
    return img