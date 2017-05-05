from __future__ import print_function
import sys

import numpy as np
from scipy.misc import imresize
from skimage.feature import hog

import utils

total_faces = 150
total_digits = 1000
face_dim = (60, 70)
digit_dim = (28, 28)

np.set_printoptions(threshold=np.inf)

class Image(object):
	def __init__(self, img):
		self.image = img
		self.image_reseize = None
		self.hog_image = None
		self.image_set = ""

	def get_feature(self):
		return self.image_resize if self.image_set in "facedata" else self.hog_image

	def generate_hog_image(self):
		# Resize the image by 400% with bicubic interpolation for better features
		self.image_resize = imresize(self.image, 400, interp='bicubic')
		fd, self.hog_image = hog(self.image_resize, visualise=True)


class ImageSet(object):
	def __init__(self, directory, file_name):
		self.images = []
		self.directory = directory
		self.file_name = file_name
		self.lenX, self.lenY = self.get_image_size()
		self.images_matrix = None

	def __len__(self):
		return len(self.images)

	def add_image(self, img):
		img.image_set = self.directory
		self.images.append(img)

	def get_images(self):
		return self.images

	def extract_image(self):
		with open("%s/%s" % (self.directory, self.file_name,), 'r') as f:
			value_map = {
				"#": 1.0,
				"+": 0.5,
			}
			current_image = []
			line_count = 1
			img_count = 1
			for line in f:
				current_image.append([value_map.get(x, 0) if x in ['#', '+'] else 0 for x in line[:-1]])

				if line_count % self.lenY == 0:
					print("Extracted image " + str(img_count), end="\r")
					sys.stdout.flush()
					img_count += 1
					image = utils.arrayToNdArray(current_image, self.lenX, self.lenY)
					if image is not None:
						yield Image(image)
					current_image = []

				line_count += 1			

	def get_image_size(self):
		return {
			"facedata": face_dim, 
			"digitdata": digit_dim,
		}.get(self.directory, (0, 0))

	def generate_vector_set(self):
		for img in self.get_images():
			yield np.ravel(img.get_feature(), order='F')

	def create_transpose_of_vectorized_images(self):
		self.images_matrix = np.ndarray(shape=(len(self), self.lenX*self.lenY*(4**2)))
		idx = 0
		for img in self.get_images():
			img_vector = np.ravel(img.get_feature(), order='F')
			self.images_matrix[idx, :] = img_vector
			idx += 1


def main():
	if len(sys.argv) != 3 and len(sys.argv) != 4:
		print("python feature_extract.py <image directory> <ascii image file>")
		return

	if sys.argv[1] not in ["facedata", "digitdata"]:
		print("Directory not supported")
		print("Must be facedata or digitdata")
		return

	image_set = ImageSet(sys.argv[1], sys.argv[2])

	# Extract images, resize, and extract hog features
	for img in image_set.extract_image():
		img.generate_hog_image()
		image_set.add_image(img)

		if len(sys.argv) == 4 and sys.argv[3] == "visualize":
			print("Beginning visualization")
			utils.display_images(img.image_resize, img.hog_image)
		break

	# Vectorize each image and add to a new matrix M
	image_set.create_transpose_of_vectorized_images()

	# Transpose the matrix M
	m = image_set.images_matrix
	return m



if __name__ == '__main__':
	main()
