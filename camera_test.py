import pco
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from PIL import Image
import glob


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


def nparray_to_video(img_lst, video_name, fps ): # img_array is a list of 3d np_arrays
	img_array = []
	for i in img_lst:
		img = cv2.cvtColor(np.array(i), cv2.COLOR_BGR2RGB)
		height, width, layers = img.shape
		size = (width, height)
		img_array.append(img)
	
	out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size) 
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()


	out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, size) 
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()



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

		plt.imshow(image, cmap='gray')
		plt.show()

def acquire_images(number_of_images, exposure_time):
	with pco.Camera() as cam:
		cam.set_exposure_time(exposure_time=exposure_time)
		cam.record(number_of_images=number_of_images)
		images = cam.images()
		cam.close()
	return images

def record_video(video_name, fps, number_of_images, exposure_time): # exposure_time is in seconds as a float or int
	images = acquire_images(number_of_images, exposure_time)
	nparray_to_video(images, video_name, fps)

if __name__ == "__main__":
	#img_lst = generate_testing_nparrays(30)
	#nparray_to_video(img_lst, 'test2.mp4', 1)

	#record_video('test.mp4', 5, 1000, 0.001) 
	record_video('test.mp4', 5, 7200 , 0.5) 