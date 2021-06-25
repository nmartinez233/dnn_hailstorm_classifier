from PIL import Image                                              
import os, sys                       

path = "../clustering/KTLX_4/cropped_pngs/"
dirs = os.listdir( path )                                       

def crop():
    for item in dirs:
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)

        # Setting the points for cropped image
        left = 64
        top = 64
        right = 192
        bottom = 192
        
        # Cropped image of above dimension
        # (It will not change original image)
        cropped = im.crop((left, top, right, bottom))
        cropped.save(f+'.png', 'png', quality=80)

crop()