"""
Example on how to plot a Skew-T plot of a sounding
--------------------------------------------------

This example shows how to make a Skew-T plot from a sounding
and calculate stability indicies.  METPy needs to be installed
in order to run this example

"""
print("Hi1")

import act
from matplotlib import pyplot as plt

try:
    import metpy
    METPY = True
    print("Hi2")
except ImportError:
    METPY = False
    print("Hi3")

if METPY:
    print("Hi4")
    # Read data
    sonde_ds = act.io.armfiles.read_netcdf("../data/sounding_data/KLOT.csv")

    # Calculate stability indicies
    sonde_ds = act.retrievals.calculate_stability_indicies(
        sonde_ds, temp_name="tdry", td_name="dp", p_name="pres")
    print("Hi")
    print(sonde_ds["lifted_index"])

    # Set up plot
    skewt = act.plotting.SkewTDisplay(sonde_ds, figsize=(15, 10))

    # Add data
    skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
    sonde_ds.close()
    plt.savefig()
