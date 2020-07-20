"""
====================================
Create a plot of NEXRAD reflectivity
====================================

An example which creates a plot containing the first collected scan from a
NEXRAD file.

"""
print(__doc__)

# Author: Jonathan J. Helmus (jhelmus@anl.gov)
# License: BSD 3 clause

import pyart
import nexradaws
import gc
import glob
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from matplotlib import use
from datetime import datetime
use('agg')

def make_radar(filename):
    try:
        radar_object = pyart.io.read(filename, include_fields=["reflectivity", "cross_correlation_ratio"])
    except:
        # Some files are corrupt, skip them
        return
    
    rad_time = datetime.strptime(radar_object.time['units'], 'seconds since %Y-%m-%dT%H:%M:%SZ')
    display = pyart.graph.RadarDisplay(radar_object)
    fig = plt.figure(figsize=(6, 5))    

    # plot super resolution reflectivity
    ax = fig.add_subplot(111)   
    display.plot('reflectivity', 0, title='NEXRAD Reflectivity',
                 vmin=-32, vmax=64, colorbar_label='', ax=ax)
    display.plot_range_ring(radar_object.range['data'][-1]/1000., ax=ax)
    display.set_limits(xlim=(-500, 500), ylim=(-500, 500), ax=ax)
    fig.savefig('%s%s' % (png_out_dir, rad_time.strftime('%Y%m%d-%H%M%S.png')), dpi=300)
    plt.close(fig)

    del radar_object.fields
    del radar_object
    del display
    del rad_time

file_directory = 'data/NEXRAD/KLOT20200417_122431_V06'
file_list = glob.glob(file_directory)
png_out_dir = 'data/'
use('agg')
i = 0
if __name__ == "__main__":
    for filename in file_list:
        make_radar(filename)
        i += 1
        if i % 10 == 0:
            print("File %d of %d processed" % (i, len(file_list)))