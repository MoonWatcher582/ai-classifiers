from scipy.misc import imresize
from skimage import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import sys

total_faces = 150
total_digits = 1000
face_dim = (60, 70)
digit_dim = (28, 28)

def arrayToNdArray(arr, lenX, lenY):
	if len(arr) < 1:
		raise IndexError
	print "Creating NdArray of size Y=%s by X=%s" % (str(len(arr)), str(len(arr[0])), )
	assert len(arr) == lenY
	assert len(arr[0]) == lenX
	return np.ndarray(shape=(len(arr), len(arr[0])), buffer=np.array(arr))

def get_hog_image(img):
	# Resize the image by 400% with bicubic interpolation for better features
	img = imresize(img, 400, interp='bicubic')
	fd, hog_image = hog(img, visualise=True)
	return hog_image, img

def display_images(original_image, hog_image):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(original_image, cmap=plt.cm.gray)
	ax1.set_title('Input Image')
	ax1.set_adjustable('box-forced')

	if hog_image is not None:
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

		ax2.axis('off')
		ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
		ax2.set_title('HOG Image')
		ax2.set_adjustable('box-forced')

	plt.show()

def extract_image(file_name, lenX, lenY):
	print "Opening file " + str(file_name)
	with open(file_name, 'r') as f:
		value_map = {
			"#": 1.0,
			"+": 0.5,
		}
		current_image = []
		line_count = 1
		img_count = 0
		for line in f:
			current_image.append([value_map.get(x, 0) if x in ['#', '+'] else 0 for x in line[:-1]])

			if line_count % lenY == 0:
				image = arrayToNdArray(current_image, lenX, lenY)
				if image is not None:
					print "Completed image " + str(img_count)
					img_count += 1
					yield image
				current_image = []

			line_count += 1			


def get_image_size(directory):
	return {
		"facedata": face_dim, 
		"digitdata": digit_dim,
	}.get(directory, (0, 0))


def main():
	if len(sys.argv) != 3 and len(sys.argv) != 4:
		print "python feature_extract.py <image directory> <ascii image file>"
		return

	lenX, lenY = get_image_size(sys.argv[1]) 
	if (lenX, lenY) not in [face_dim, digit_dim]:
		print "Directory not recognized"
		return

	images = []
	hog_images = []

	# Extract images, resize, and extract hog features
	print "File is %s/%s, whose images are %s by %s" % (sys.argv[1], sys.argv[2], str(lenX), str(lenY), )
	for img in extract_image("%s/%s" % (sys.argv[1], sys.argv[2]), lenX, lenY):
		print "Extracting HOG features"
		hog_img, img = get_hog_image(img)
		images.append(img)
		hog_images.append(hog_img)

		if len(sys.argv) == 4 and sys.argv[3] == "visualize":
			print "Beginning visualization"
			display_images(img, hog_img)

	# Vectorize each image and add to a new matrix M
	print "Selecting image set to vectorize"
	print "Selecting " + ("images for faces" if "face" in sys.argv[1] else "hog images for digits")
	feature_images = images if "face" in sys.argv[1] else hog_images

	print "Creating transposed matrix of vectorized images"
	images_matrix = np.ndarray(shape=(len(feature_images), lenX*lenY*(4**2)))
	idx = 0
	for img in feature_images:
		img_vector = np.ravel(img, order='F')
		images_matrix[idx, :] = img_vector
		idx += 1

	# Transpose the matrix M
	m = images_matrix
	print m
	return m



if __name__ == '__main__':
	main()
