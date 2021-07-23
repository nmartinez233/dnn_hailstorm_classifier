from PIL import Image                                              
import os, sys                       

path = "../clustering/KLZK_4/test/"
dirs = os.listdir( path )                                       

def crop():
    for item in dirs:
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)

        # Setting the points for cropped image
        left = 128
        top = 128
        right = 384
        bottom = 384
        
        # Cropped image of above dimension
        # (It will not change original image)
        cropped = im.crop((left, top, right, bottom))
        cropped.save(f+'.png', 'png', quality=80)

crop()