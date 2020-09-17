import shutil, glob

png_directory = "../clustering/DenseNet_4/cluster3/*png"
png_list = glob.glob(png_directory)

nexrad_directory = '../data/NEXRAD/'

copy_to_directory = "../data/used_NEXRAD_data/cluster3/"

def find_v06(file_name):
    inter_name = "KLOT"+file_name[34:(len(file_name)-4)]+"_V06"
    print(inter_name)
    v06_name = inter_name[:len(inter_name)-11]+"_"+inter_name[len(inter_name)-10:]
    print(v06_name)
    full_name = nexrad_directory+v06_name
    print(full_name)

    full_copy = copy_to_directory+v06_name

    print(full_copy)
    shutil.copy2(full_name, full_copy)

if __name__ == "__main__":
    for file_name in png_list:
        find_v06(file_name)

    