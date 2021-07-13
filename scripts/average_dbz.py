import glob
import pyart
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use
from datetime import datetime

def get_dbz(filename):
    try:
        radar_object = pyart.io.read(filename, include_fields=["reflectivity", "cross_correlation_ratio"])
    except:
        # Some files are corrupt, skip them
        return

    bottom_sweep = radar_object.extract_sweeps([0])
    reflectivity = bottom_sweep.fields['reflectivity']['data']
    new_dbz = np.add(sum_dbz, reflectivity)
    return new_dbz

def plot_dbz(average_dbz, filename):
    radar = pyart.io.read(filename)
    rad_time = datetime.strptime(radar.time['units'], 'seconds since %Y-%m-%dT%H:%M:%SZ')
    print(average_dbz.shape)
    mask_dict = {'data': average_dbz, 'units': 'dBZ', 'long_name': 'reflectivity_mask',
     '_FillValue': average_dbz.fill_value, 'standard_name': 'reflectivity_mask'}
    # Adding this field into the radar object using radar.add_field()
    radar.add_field('reflectivity_mask', mask_dict, replace_existing=True)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    display = pyart.graph.RadarDisplay(radar)
    display.plot_ppi('reflectivity_mask', sweep=0, vmin=-32, vmax=64, 
                     ax=ax, fig=fig, title='', colorbar_flag=False, cmap='pyart_HomeyerRainbow')
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.axis('off')
    fig.savefig('%s%s' % (png_out_dir, rad_time.strftime('%Y%m%d-%H%M%S.png')), dpi=300)

    plt.show()


png_out_dir = '../'
file_directory = '../data/used_NEXRAD_data/cluster0/*V06*'
file_list = glob.glob(file_directory)
sum_dbz = np.zeros((720,1832))
i = 0

if __name__ == "__main__":
    for filename in file_list:
        sum_dbz = get_dbz(filename)
        temp = filename
        i += 1
        if i % 10 == 0:
            print("File %d of %d processed" % (i, len(file_list)))
    average_dbz = np.divide(sum_dbz, len(file_list))
    print(average_dbz)
    ma_average_dbz = np.ma.where(average_dbz > 0, 1, 0)
    print(ma_average_dbz)
    plot_dbz(ma_average_dbz, temp)
    