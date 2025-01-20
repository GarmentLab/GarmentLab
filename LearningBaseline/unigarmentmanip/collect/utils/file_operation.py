import os
import numpy as np

def get_unique_filename(base_filename, extension=".png"):
    counter = 0
    filename = f"{base_filename}_{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}_{counter}{extension}"
        
    if extension==".ply":
        return filename,counter
    return filename