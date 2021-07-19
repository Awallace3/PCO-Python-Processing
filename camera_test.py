from numpy.core.fromnumeric import size
import pco
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from PIL import Image
import glob
import os
import subprocess
from sys import argv

def generate_testing_nparrays(size=10):
	imgb = np.zeros([600, 600, 3], dtype=np.uint8)
	imgb.fill(255)
	imgw = np.zeros([600, 600, 3], dtype=np.uint8)
	imgw.fill(0)
	img_lst = [imgb, imgw]
	for i in range(100):
		if i % 2 ==0:
			img_lst.append(imgb)
		else:
			img_lst.append(imgw)
	return img_lst

def send_3d_to_file(img_lst):
	os.remove('storage.txt')
	f = open("storage.txt", 'w')
	f.write("\n")
	f.close()
	with open("storage.txt", 'ab') as f:
		for i in img_lst:
			i = i.reshape(i.shape[0], -1)
			f.write(b"\n")
			np.savetxt(f,i)

def gen_3d_from_file():
	cmd = "grep -n ^$ storage.txt > breaks.txt"
	subprocess.call(cmd, shell=True)
	with open("breaks.txt", 'r') as fp:
		data = fp.readlines()
	print(data)
	#img_lst = []
	#loaded_arr = np.genfromtxt('storage.txt', skip_header=num1, skip_footer=num2)
	#load_original_arr = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])
  

def take_pictures(img_lst):
	start_time = time.time()
	for n, i in enumerate(img_lst):
		img = Image.fromarray(i.astype(np.uint8))
		img.save('images/%d_%04d.png' % (start_time, n))


def nparray_to_video(img_lst, video_name, fps ): # img_array is a list of 3d np_arrays

	#img_array = []
	start_time = time.time()
	for n, i in enumerate(img_lst):

		#print("img1:",i)
		#cv2.imwrite("images/image.png", i)
		img = Image.fromarray(i.astype(np.uint8))

		img.save('images/%d_%04d.png' % (start_time, n))
		#img = cv2.imread("images/image.png", cv2.IMREAD_GRAYSCALE)
	
	"""
		print("img2:", img)
		height, width = i.shape
		size = (width, height)
		img_array.append(img)
	

	out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size) 
	for i in range(len(img_array)):
		print("img3:",img_array[i])
		out.write(img_array[i])
	out.release()
	

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	writer = cv2.VideoWriter(video_name, fourcc, fps, size)

	for frame in img_lst:
		writer.write(frame)

	writer.release() 
	"""



def test_image_to_video():
	img_array = []
	for filename in glob.glob("images/*.png"):
		img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
		height, width, layers = img.shape
		size = (width, height)
		img_array.append(img)
	print(img_array)
	out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

def test_take_picture():
	with pco.Camera() as cam:
		cam.record()
		image, meta = cam.image()

		plt.imshow(image)
		plt.show()

def acquire_images(number_of_images, exposure_time):
	with pco.Camera() as cam:
		cam.set_exposure_time(exposure_time=exposure_time)
		cam.record(number_of_images=number_of_images)
		images = cam.images()
		return images[0]

def record_video(video_name, fps, number_of_images, exposure_time): # exposure_time is in seconds as a float or int
	
	images = acquire_images(number_of_images, exposure_time)
	nparray_to_video(images, video_name, fps)

def acquire_images(exposure_time, time_limit):
	with pco.Camera() as cam:
		cam.set_exposure_time(exposure_time=exposure_time)
		start_time = time.time()
		for i in range(time_limit):
				
			cam.record(1)
			image, meta = cam.image()
			img = Image.fromarray(image.astype(np.uint8))
			img.save('images/%d_%04d.png' % (start_time, i))
			

	return 


if __name__ == "__main__":
	# uncomment these lines for testing cv2
	#img_lst = generate_testing_nparrays(30)
	#nparray_to_video(img_lst, 'test2.mp4', 1)

	# storing 3d numpy arrays in file 
	#send_3d_to_file(img_lst)
	#gen_3d_from_file()
	
	# testing procedure
	#record_video('test.mp4', 5, 1000, 0.001) 
	"""
	record_video('test.mp4', 1, 1000, 2) 
	time_limit = 100 # hours
	for i in range(time_limit):
		record_video
	
	"""
	time_limit = 1000
	if len(argv) > 1:
		time_limit=int(argv[1])
	acquire_images(1, time_limit)
	#acquire_images_time(1000, 0.01)
	#test_take_picture()