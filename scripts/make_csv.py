#create a function that outputs the percentage of DBZ >20
#save the output to a csv file

import pyart
import glob
import numpy as np
import pandas as pd

file_directory = '../data/NEXRAD2/*V06*'
file_list = glob.glob(file_directory)
csv_directory = '../data/reflectivity2.csv'
i = 0
array = []

def find_percent(filename):
    try:
        radar_object = pyart.io.read(filename, include_fields=["reflectivity"])
    except:
        # Some files are corrupt, skip them
        return
    bottom_sweep = radar_object.extract_sweeps([0])
    total = bottom_sweep.fields['reflectivity']['data']
    gt20 = np.extract(total > 20, total)
    len_gt20 = gt20.size
    len_total = total.size
    pcoverage = 100*(len_gt20/len_total)


    array.append(pcoverage)

    del radar_object.fields
    del radar_object
    del bottom_sweep
    del total
    del gt20
    del len_gt20
    del len_total
    del pcoverage
    



if __name__ == "__main__":
    for filename in file_list:
        find_percent(filename)
        i += 1
        if i % 10 == 0:
            print("File %d of %d processed" % (i, len(file_list)))

    
    df = pd.DataFrame(array)
    del array
    df.to_csv(csv_directory, index=False)
    