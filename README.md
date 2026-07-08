# pyCRED
Data acquisition and analysis to use and characterise the C-RED ONE camera in hq lab. 
All following instructions and uploaded operation scripts are relative to pathnames and directories on gpupc. This will be made system and directory agnostic in the future.

# Steps to start-up cooling and camera (in order)

## Chiller

Depending on the chiller used, follow instructions either from the OCAM2k short instructions or the HQ Lab Chiller / OCAM flow protection document

## Camera

1. Connect the camera link cables to the C-RED one as follows: the cable marked blue goes to CL1, and the other end goes to the top port (also marked in blue) of the GPU
2. Connect the LEMO power cable to the power supply of the camera, which should be connected to the wall socket
3. Connect to gpupc:3 on VNC viewer on your laptop
4. Right click on the desktop and open a terminal
5. Run `cd/opt/EDTpdv`
6. Once you’re in the EDT directory, run the configuration file to set the frame grabber to capture images ` ./initcam -f ~/../../usr/local/aodev/CRED-One/cred_default.cfg `
7. Run `./serial_cmd ‘temperature'` to get the current temperature
8. If this returns a garbage output, swap the cameralink cables on the gpu side.
9. Then run `./serial_cmd ‘temperature’` again to verify
10. Run `./serial_cmd status` to get the camera status
12. If it shows ready, then the camera is ready to be cooled. Run `./serial_cmd ‘set cooling on’` to start the cooling. For other status and their meanings, refer the C-RED ONE User Manual.
13. Occasionally check the temperatures. The camera needs to be cooled to about 80K. The relevant value is denoted next to cryopt(pulseTube): in the `./serial_cmd ‘temperature’` output. And to check if the camera is being cooled, run `./serial_cmd status` and it should return `isbeingcooled`.
14. When the cryod and cryopt temperatures have dropped below 90K, check the camera status, by running `./serial_cmd ‘status’`. If it says operational, it means the camera is ready to take images.


## Take Images
1. For a live view, run: `kpython3 CREDOne_Disp.py`
2. To take an image using frame grabber commands: `./take -N 200 -f \'{filename}\'` from `~/opt/EDTpdv`
