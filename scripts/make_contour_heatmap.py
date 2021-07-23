import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import xarray as xr
import cartopy
import pyart
import glob

def sum_reflectivity(filename):
    try:
        data = netCDF4.Dataset(filename)
    except:
        # Some files are corrupt, skip them
        return

    x = np.squeeze(np.array(data['reflectivity'][:]))
    temp = x[0][:][:]

    reflectivity = np.dstack((sum_reflectivity_array, temp))
    
    del temp
    del x
    return reflectivity

def make_heatmap(mean_reflectivity):
    x = np.linspace(-150, 150, num=600)
    y = np.linspace(-150, 150, num=600)

    plt.contourf(x, y, mean_reflectivity, vmin=-10, vmax=30, 
        cmap='pyart_HomeyerRainbow')


    plt.xlabel("km East of Radar") 
    plt.ylabel("km North of Radar")
    plt.colorbar()
    plt.title(cluster_name+" Heatmap")
    plt.savefig('%s%s' % (png_out_dir, cluster))
    return


data_directory = '../data/KLZK_NETCDF4/cluster0/*nc*'
data_list = glob.glob(data_directory)
png_out_dir = '../clustering/KLZK_4/KTLX_trained/contour/'
sum_reflectivity_array = np.zeros((600,600, 1))
cluster = "cluster0.png"
cluster_name = 'cluster0'

i = 0

if __name__ == "__main__":
    for filename in data_list:
        sum_reflectivity_array = sum_reflectivity(filename)
        i+=1
        if i % 10 == 0:
            print("File %d of %d processed" % (i, len(data_list)))

    mask = np.ma.where(sum_reflectivity_array > -20, sum_reflectivity_array, np.nan)
    masked_mean = np.ma.mean(mask, axis=2)

    print('Mean Reflectivity complete, generating image...')
    make_heatmap(masked_mean)