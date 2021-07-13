import pyart
import nexradaws
import gc
import glob
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from distributed import Client, wait
from dask_jobqueue import SLURMCluster
from matplotlib import use
from datetime import datetime
use('agg')


def make_plot(filename):
    try:
        radar_object = pyart.io.read(filename, include_fields=["reflectivity", "cross_correlation_ratio"])
    except:
        # Some files are corrupt, skip them
        return

    bottom_sweep = radar_object.extract_sweeps([0])
    total = bottom_sweep.fields['reflectivity']['data']
    gt20 = np.extract(total > 20, total)
    len_gt20 = gt20.size
    len_total = total.size
    pcoverage = len_gt20/len_total

    if pcoverage == 1.:
        decision = 1
    else:
        decision = np.random.choice([0,1], p=[1.-pcoverage, pcoverage])

    if decision == 1:
        ## Parse the scan time
        rad_time = datetime.strptime(radar_object.time['units'], 'seconds since %Y-%m-%dT%H:%M:%SZ')

        # Apply mask here
        mask = np.where(np.logical_and(radar_object.fields['reflectivity']['data'] > 20, 
                        radar_object.fields['cross_correlation_ratio']['data'] > 0.98), 
                        radar_object.fields['reflectivity']['data'], np.nan)
        radar_object.add_field_like('reflectivity', 'storm_mask', mask, replace_existing=True)
    
        # Create the image for DNN
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        display = pyart.graph.RadarDisplay(radar_object)
        display.plot_ppi('storm_mask', sweep=0, vmin=-32, vmax=64, 
                         ax=ax, fig=fig, title='', colorbar_flag=False, cmap='pyart_HomeyerRainbow')
        ax.set_xlim([-300, 300])
        ax.set_ylim([-300, 300])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.axis('off')
        fig.savefig('%s%s' % (png_out_dir, rad_time.strftime('%Y%m%d-%H%M%S.png')), dpi=300)
        plt.close(fig)
        print(filename, "processed!")
        print("Generated png for time %s" %  rad_time.strftime('%Y%m%d-%H%M%S.png'))
        # Free memory
        del radar_object.fields
        del radar_object
        del display
        del mask
        del rad_time
        del pcoverage
        del bottom_sweep
        del total 
        del gt20

        return

    else:
        print(filename, "skipped!")

        del radar_object.fields
        del radar_object
        del pcoverage
        del bottom_sweep
        del total 
        del gt20

        return #do not create image

file_directory = '../data/KTLX_data/*'
file_list = glob.glob(file_directory)
print(file_list)
png_out_dir = '../data/KTLX_pngs/'
i = 0
use('agg')

if __name__ == "__main__":
    #for filename in file_list:
    #    make_plot(filename)
    #    i += 1
    #    if i % 10 == 0:
    #        print("File %d of %d processed" % (i, len(file_list)))
    Cluster = SLURMCluster(processes=6, cores=36, memory='128GB', walltime='2:00:00')
    Cluster.scale(36)
    client = Client(Cluster)
    print("Waiting for workers...")
    while(len(client.scheduler_info()["workers"]) < 6):
        i = 1
    futures = client.map(make_plot, file_list)
    wait(futures)
    client.close() 
