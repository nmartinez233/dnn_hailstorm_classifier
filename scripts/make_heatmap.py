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
    row_idx = np.array(np.arange(100,300))
    col_idx = np.array(np.arange(100,300))

    tmp = x[0][:][:]
    temp = tmp[row_idx[:, None], col_idx]

    reflectivity = np.dstack((sum_reflectivity_array, temp))
    del tmp
    del temp
    del row_idx
    del col_idx
    del x
    return reflectivity

def make_heatmap(mean_reflectivity):
    plt.pcolormesh(mean_reflectivity, vmin=-32, vmax=64, 
        cmap='pyart_HomeyerRainbow')
    plt.title(cluster_name+" Heatmap")
    plt.colorbar()
    plt.savefig('%s%s' % (png_out_dir, cluster))
    return


data_directory = '../data/netCDF4/cluster3/*nc*'
data_list = glob.glob(data_directory)
png_out_dir = '../data/netCDF4/'
sum_reflectivity_array = np.zeros((200,200, 1))
cluster = "cluster3.png"
cluster_name = 'cluster3'

i = 0

if __name__ == "__main__":
    for filename in data_list:
        sum_reflectivity_array = sum_reflectivity(filename)
        i+=1
        if i % 10 == 0:
            print("File %d of %d processed" % (i, len(data_list)))

    mask = np.ma.where(sum_reflectivity_array > -20, sum_reflectivity_array, np.nan)
    masked_mean = np.ma.mean(mask, axis=2)

    print(masked_mean)
    print('Mean Reflectivity complete, generating image...')
    make_heatmap(masked_mean)