from PIL import Image                                              
import os, sys                       

path = "../data/images/256_pngs/"
dirs = os.listdir( path )                                       

def resize():
    for item in dirs:
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)
        imResize = im.resize((256,256), Image.ANTIALIAS)
        imResize.save(f+'.png', 'png', quality=80)

resize()