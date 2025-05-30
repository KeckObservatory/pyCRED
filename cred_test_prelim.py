############################## Import Libraries ###############################

## Math Library
import numpy as np
## Subprocess library
import subprocess
## System library
import sys
## Operating system library
import os
## PIL library used to read bitmap files
from PIL import Image
## Library used to plot graphics and show images
import matplotlib.pyplot as plt
## Time library 
import time
## Datatime Library
from datetime import datetime
## import library used to manipulate fits files
from astropy.io import fits
## to sort
from glob import glob

############################## Local Definitions ##############################

def get_image():
	"""
	Function used to get images from the EDT framegrabber.
	"""
	# Location of the bitmap frames (temporary)
	filename = '/home/aodev/CRED-One/Data/tmp/CRED_frame.raw'
	# Prepare command to pull an image
	command  = f'cd /opt/EDTpdv/; ./take -N 200 -f \'{filename}\''
	# Starting image acq
	# print('Starting image acquistion')
	# Pull an image
	trash = subprocess.run(command, shell = True, capture_output=True, text=True)
	# print("STDOUT:", trash.stdout) # print this no matter what 
	# save a timestamp
	now = datetime.now()
	time_str = now.strftime("%Y%m%d_%H%M%S")   
	# Open the file and convert it into a numpy array
	im = np.fromfile(filename, dtype=np.uint16)
	im_arr = im.reshape((256, 320))
	return im_arr, time_str

def get_image_multiframe(nframes, show_output=False, clear_directory=False):
	"""
	Captures multiple image frames and returns them as a numpy array.
	"""
	# Location of the frames (temporary)
	filename = '/home/aodev/CRED-One/Data/tmp/CRED_frame'
	# **Clear existing files if requested**
	if clear_directory:
		for f in glob(filename+'_*'):
			os.remove(f)
		print("Deleted all previous image files.")
	# Prepare command to pull an image
	im_arr_big = np.zeros((nframes,256,320))
	for i in range(nframes):
		im_arr, time_str = get_image()
		im_arr_big[i] = im_arr
	return im_arr_big


def save_images_legacy(filename, frames, fps):
	"""
	Function to save images (legacy)
	"""
	# Set frame rate
	command = f'cd /opt/EDTpdv/; ./serial_cmd \'set fps {fps}\''
	print(command)
	trash = subprocess.run(command, shell = True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	print("STDERR:", trash.stderr)
	# Check if the command was successful
	if trash.returncode == 0:
		print("Command executed successfully!")
	else:
		print(f"Command failed with return code {trash.returncode}")
	new_cmd = f'cd /opt/EDTpdv/; ./serial_cmd \'fps\''
	trash = subprocess.run(new_cmd, shell = True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# Initialise cube:
	im_arr = np.zeros((frames,256,320))
	# Start loop
	for i in range(frames):
		# Get the image
		im, time_str = get_image()
		im_arr[i] = im
	# Get the median
	med_cube = np.median(im_arr, axis = 0)
	# save the data cubes
	date = datetime.now().strftime("%Y%m%d")
	# make directory if it doesn't exist
	im_dir = f'/home/aodev/Data/{date}'
	os.makedirs(im_dir, exist_ok=True)
	f_name = os.path.join(im_dir,filename+'.fits')
	fits.writeto(f_name,im_arr,overwrite=True)
	print(f'Saved cube as {f_name}')
	f_med_name = os.path.join(im_dir,filename+'_med.fits')
	fits.writeto(f_med_name,med_cube,overwrite=True)
	print(f'Saved cube as {f_med_name}')


def test_gains(gains,im_type, show_output = False):
	"""
	Function to get images of varying gains
	"""
	for i in gains:
		date = datetime.now().strftime("%y%m%d")
		# make directory if it doesn't exist
		im_dir = f'/home/aodev/CRED-One/Data/{date}/{im_type}'
		os.makedirs(im_dir, exist_ok=True)
		filename = im_dir+f'/gain_{i}'
		# set gain
		gain_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set gain {i}'"
		trash = subprocess.run(gain_cmd, shell = True, capture_output=True, text=True)
		if show_output:
			print("STDOUT:", trash.stdout)
			print("STDERR:", trash.stderr)
		check_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'gain'"
		trash = subprocess.run(check_cmd, shell = True, capture_output=True, text=True)
		print("STDOUT:", trash.stdout) # print this no matter what
		## Check if the command was successful
		# if trash.returncode == 0:
		#	print("Command executed successfully!")
		# else:
		#	print(f"Command failed with return code {trash.returncode}")
		#	break
		# capture image
		if show_output:
			ims = get_image_multiframe(50, show_output=True, clear_directory=True)
		else:
			ims = get_image_multiframe(50, clear_directory=True)
		# med_ims = np.median(ims, axis=0).astype(np.uint8)  # Convert to uint8
		# save as fits
		# fits.writeto(filename+'.fits',ims, overwrite=True)
		np.save(filename+'.npy', ims)
		
	print('Done!')


def darks_fps(fps_list):
	"""
	Function to get darks at different frame rates
	"""
	for i in fps_list:
		date = datetime.now().strftime("%y%m%d")
		# make directory if it doesn't exist
		im_dir = f'/home/aodev/CRED-One/Data/{date}/darks_grs/fps{i:.2f}'
		os.makedirs(im_dir, exist_ok=True)
		filename = im_dir+f'/dark_fps{i:.2f}_'
		# set fps
		fps_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set fps {i}'"
		trash = subprocess.run(fps_cmd, shell = True, capture_output=True, text=True)
		print("STDOUT:", trash.stdout)
		check_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'fps'"
		trash = subprocess.run(check_cmd, shell = True, capture_output=True, text=True)
		print("STDOUT:", trash.stdout) # print this no matter what
		# capture images
		img_cmd = f"cd /opt/EDTpdv/; ./take -N 200 -l 1000 -f '{filename}'"
		print(img_cmd)
		trash = subprocess.run(img_cmd, shell = True, capture_output=True, text=True)
		print("STDOUT:", trash.stdout)
	print('Done!')


def dark_bursts(fps, ndr):
	"""
	Function to get burst sequences of darks
	"""
	date = datetime.now().strftime("%y%m%d")
	# make directory if it doesn't exist
	im_dir = f'/home/aodev/CRED-One/Data/{date}/darks_ndr/fps{fps:.0f}_ndr{ndr:.0f}'
	os.makedirs(im_dir, exist_ok=True)
	filename = im_dir+f'/darks_fps{fps:.0f}_ndr{ndr:.0f}_'
	# set global reset mode
	burst_mode=  f"cd /opt/EDTpdv/; ./serial_cmd 'set mode globalresetbursts'"
	trash = subprocess.run(burst_mode, shell = True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# set raw images on
	raw_images = f"cd /opt/EDTpdv/; ./serial_cmd 'set rawimages on'"
	trash = subprocess.run(raw_images, shell = True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# set ndr
	ndr_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set nbreadworeset {ndr}'"
	trash = subprocess.run(ndr_cmd, shell = True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# set fps
	fps_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set fps {fps}'"
	trash = subprocess.run(fps_cmd, shell = True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# insert wait
	time.sleep(70)
	# capture images
	img_cmd = f"cd /opt/EDTpdv/; ./take -N 200 -l 5000 -f '{filename}'"
	print(img_cmd)
	trash = subprocess.run(img_cmd, shell = True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	print('Done!')

def dark_bursts_gain(gain, fps, ndr):
	"""
	Function to get burst sequences of darks
	"""
	# set global reset mode
	burst_mode = "cd /opt/EDTpdv/; ./serial_cmd 'set mode globalresetbursts'"
	trash = subprocess.run(burst_mode, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# set gain
	gain_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set gain {gain}'"
	trash = subprocess.run(gain_cmd, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# print gain
	check_cmd = "cd /opt/EDTpdv/; ./serial_cmd 'gain'"
	trash = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)  # print this no matter what
	# set gain twice
	gain_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set gain {gain}'"
	trash = subprocess.run(gain_cmd, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# print gain
	check_cmd = "cd /opt/EDTpdv/; ./serial_cmd 'gain'"
	trash = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)  # print this no matter what
	# set raw images on
	raw_images = "cd /opt/EDTpdv/; ./serial_cmd 'set rawimages on'"
	trash = subprocess.run(raw_images, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# set ndr
	ndr_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set nbreadworeset {ndr}'"
	trash = subprocess.run(ndr_cmd, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# set fps
	fps_cmd = f"cd /opt/EDTpdv/; ./serial_cmd 'set fps {fps}'"
	trash = subprocess.run(fps_cmd, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# insert wait
	time.sleep(30)
	# capture images
	filename = '/home/aodev/CRED-One/Data/tmp/select/CRED_frame_'
	img_cmd = f"cd /opt/EDTpdv/; ./take -N 250 -l 2000 -f '{filename}'"
	trash = subprocess.run(img_cmd, shell=True, capture_output=True, text=True)
	print("STDOUT:", trash.stdout)
	# convert to numpy array
	im_arr = np.zeros((2000, 256, 320))
	for i in range(2000):
		im = np.fromfile(filename+f'{i:04d}.raw', dtype=np.uint16)
		im_arr[i] = im.reshape((256, 320))
	print("STDOUT:", trash.stdout)
	print('Done!')
	return im_arr

def test_dark_gains(gain_arr, fps_arr, ndr_arr):
	"""
	Test if dark current varies with gain
	"""
	for gain in gain_arr:
		for fps in fps_arr:
			for ndr in ndr_arr:
				date = datetime.now().strftime("%y%m%d")
				# make directory if it doesn't exist
				im_dir = f'/home/aodev/CRED-One/Data/{date}/darks_gains'
				os.makedirs(im_dir, exist_ok=True)
				filename = im_dir+f'/darks_gain{gain:.0f}_fps{fps:.0f}_ndr{ndr:.0f}'
				# take image
				im_arr = dark_bursts_gain(gain, fps, ndr)
				# write file
				np.save(filename+'.npy', im_arr)
				# Print done message
				print(f'Done with gain = {gain:.0f}, fps = {fps:.0f} and ndr = {ndr:.0f}')


	
	
