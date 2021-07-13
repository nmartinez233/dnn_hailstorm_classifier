import matplotlib.pyplot as plt
from distributed import Client, wait
from dask_jobqueue import SLURMCluster
import numpy as np
import pyart
import glob

def convert(file_name):
    radar = pyart.io.read(file_name)

    grid = pyart.map.grid_from_radars(radar, grid_shape=(10, 600, 600),
        grid_limits=((0.,3000,), (-150000., 150000.), (-150000., 150000.)))
    
    filename = grid_file_directory+file_name[31:(len(file_name)-4)]+".nc"
    print("Writing",filename)
    pyart.io.write_grid(filename, grid, format='NETCDF4',
        write_point_lon_lat_alt=True)
    print(filename,"processed!")

i = 0
j = 0

if __name__ == "__main__":
    for j in range(3,4):
        i = 0
        file_directory = '../data/used_KTLX_data/cluster%d/*V06*' % (j)
        file_list = glob.glob(file_directory)
        grid_file_directory = '../data/KTLX_NETCDF4/cluster%d' % (j)
        
        Cluster = SLURMCluster(processes=6, cores=36, memory='128GB', walltime='2:00:00')
        Cluster.scale(36)
        client = Client(Cluster)
        print("Waiting for workers...")
        while(len(client.scheduler_info()["workers"]) < 6):
            i = 1
        futures = client.map(convert, file_list)
        wait(futures)
        client.close() 

        """
        for filename in file_list:
            convert(filename)
            i += 1
            #if i % 10 == 0:
            print("File %d of %d processed" % (i, len(file_list)))"""
        