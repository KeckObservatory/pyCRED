#!/usr/bin/env python3
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cred_test_prelim as cred

# Load background image or set to zeros if missing
try:
    bkgd = fits.open('/usr/local/aodev/Data/220201/OCAM2/bkg_g300_med.fit')[0].data
    print('bkgd loaded')
except:
    bkgd = np.zeros([240, 240])

def update_display():
    """Function to update image display in a loop."""
    while plt.fignum_exists(fig.number):  # Check if the figure is still open
        img_data = cred.get_image()[0].astype(np.float32)
        img_display.set_data(img_data)
        fig.canvas.flush_events()  # Process GUI events (avoids freezing)
        time.sleep(0.1)  # Small delay to allow real-time updates

if __name__ == '__main__':
    inter = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Set up Matplotlib figure
    plt.ion()  # Interactive mode ON
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.zeros((240, 240)), cmap='magma', vmin=19500, vmax=25500,origin='lower')
    plt.colorbar(img_display)

    try:
        update_display()
    except KeyboardInterrupt:
        print("\nLive display stopped by user.")
    finally:
        plt.ioff()
        plt.close(fig)
