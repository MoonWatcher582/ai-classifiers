import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

def arrayToNdArray(arr, lenX, lenY):
	if len(arr) < 1:
		raise IndexError
	print "Creating NdArray of size Y=%s by X=%s" % (str(len(arr)), str(len(arr[0])), )
	assert len(arr) == lenY
	assert len(arr[0]) == lenX
	return np.ndarray(shape=(len(arr), len(arr[0])), buffer=np.array(arr))

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

