from PIL import Image
import numpy as np

def image_to_array(img):
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = img.convert("L")
    
    img_array = np.asarray(img)
    
    img_array[img_array < 128] = 1
    img_array[img_array >= 128] = 0

    img_vector = img_array.flatten()
    
    return img_vector