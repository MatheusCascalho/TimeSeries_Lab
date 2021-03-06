import os
from shutil import copyfile
import pandas as pd
import numpy as np
from PIL import Image


# file = '/home/matheus_cascalho/Documents/Matheus Cascalho/MINDS/TimeSeries_Lab/sky.jpeg'
# print(os.path.getmtime(file))

def rename_images(path: str, to_dir: str = None, extension: str = "png") -> None:
    dir = [img for img in os.listdir(path) if img.endswith(extension)]
    for file in dir:
        year = file[:4]
        month = file[4:6]
        day = file[6:8]
        hour = file[8:10]
        min = file[10:12]
        name = f'{year}_{month}_{day} {hour}:{min}.{extension}'
        if to_dir is not None:
            copyfile(path + '/' + file,
                     to_dir + '/' + name)
        else:
            os.rename(path + '/' + file,
                      path + '/' + name)


def dir_to_data(path: str,
                extension: str = 'jpg',
                qtt: int = 10,
                convert_type: str = 'RGB') -> list:
    files = [img for img in os.listdir(path) if img.endswith(extension)]
    if len(files) >= qtt:
        files = files[:qtt]
    data_images = []
    for file in files:
        filepath = path + '/' + file
        img = Image.open(filepath)
        data = np.asarray(img.convert(convert_type))
        data_images.append(data)
    return data_images


if __name__ == '__main__':
    path = 'images/20201019'
    dst = 'images/dataset'
    # rename_images(path, to_dir=dst, extension='raw.jpg')
    dataset = dir_to_data(dst, qtt=3)
    print(dataset[2])
