from skimage import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import sys

def arrayToNdArray(arr):
	if len(arr) < 1:
		raise IndexError
	print "Creating NdArray of size Y=" + str(len(arr)) + " by X=" + str(len(arr[0]))
	return np.ndarray(shape=(len(arr), len(arr[0])), buffer=np.array(arr))

def get_hog_image(img):
	fd, hog_image = hog(img, visualise=True)
	return hog_image

def display_images(original_image, hog_image):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(original_image, cmap=plt.cm.gray)
	ax1.set_title('Input Image')
	ax1.set_adjustable('box-forced')

	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('HOG Image')
	ax2.set_adjustable('box-forced')

	plt.show()

def extract_image(file_name):
	print "Opening file " + str(file_name)
	with open(file_name, 'r') as f:
		current_image = []
		newline_count = 0
		img_count = 0
		for line in f:
			if line.strip() == "":
				newline_count += 1
			else:
				newline_count = 0

			if newline_count >= 2 and len(current_image) > 0:
				print "Completed image " + str(img_count)
				yield arrayToNdArray(current_image)
				current_image = []
				img_count += 1
			
			current_image.append([1. if x == '#' else 0. for x in line[:-1]])


def main():
	if len(sys.argv) != 2:
		print "python feature_extract.py <ascii image file>"

	images = []
	for img in extract_image(sys.argv[1]):
		images.append(img)

	print "Extracting HOG features"
	hog_img = get_hog_image(images[0])

	print "Beginning visualization"
	display_images(images[0], hog_img)

if __name__ == '__main__':
	main()
