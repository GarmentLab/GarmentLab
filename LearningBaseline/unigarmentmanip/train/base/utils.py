import os
import sys

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('Path already exists.')
        raise ValueError
    return path

def force_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('Path already exists.')
        os.makedirs(path,exist_ok=True)
    return path