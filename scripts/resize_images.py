from PIL import Image                                              
import os, sys                       

path = "../data/images/smaller_filtered_pngs/"
dirs = os.listdir( path )                                       

def resize():
    for item in dirs:
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)
        imResize = im.resize((200,200), Image.ANTIALIAS)
        imResize.save(f+'.png', 'png', quality=80)

resize()