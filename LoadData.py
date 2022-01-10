"""
Author email : zgzhang@whu.edu.cn
Creation date: 2021.11.24
File name    : LoadData.py
"""
import os
import numpy as np

def loadData(path):
    def loadImg(path):
        img = []

        with open(path, "r") as f:
            for row in f.readlines():
                if row.strip():
                    img += [int(i) for i in row.strip()]
        return img
    
    files = os.listdir(path)
    img = np.zeros([len(files), 32*32], dtype=np.int)
    lable = np.zeros([len(files)], dtype=np.int)
    
    for i, fileName in enumerate(files):

        lable[i] = int(fileName.split("_")[0])

        img[i] = loadImg(path+"/"+fileName)

    return img, lable