import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filter import threshold_otsu, threshold_adaptive
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square, disk
from skimage.measure import regionprops
from skimage.color import label2rgb, rgb2hed
from skimage.io import imread, imshow, imsave
from skimage import filter
from skimage.exposure import rescale_intensity
from skimage.filter.rank import entropy

def loadshow_img(filename):
	"""
	This function loads an image from the disk returns the matrix
	representation, and displays it on screen.

	Inputs:
	- filename: the name of the file to open.

	Outputs:
	- image: a matrix describing the original image file.

	To screen: a visual representation of the image file.
	"""
	image = imread(filename)
	imshow(image)
	return image

def showchannel(image, channel):
	"""
	This function takes an "image" (not a filename) returned from
	loadshow_img() and returns the specified channel.

	Inputs:
	- image: a matrix describing an image in RGB
	- channel: the channel that you wish to display
		- 0 = red 
		- 1 = green 
		- 2 = blue

	Outputs:
	- channel: a matrix describing the image in R, G or B.

	To screen: a representation of the particular channel selected.
	"""
	channel = image[:,:,channel]
	imshow(channel)
	return channel

def threshold_image(image, threshold=0):
	"""
	This function takes out any values in an image's RGB matrix that are
	below the threshold value.

	Inputs:
	- image: a matrix describing an image with only one channel represented.
	- threshold: a value, between 0 and 1, for which if an image matrix's
				 value is below, will be set to 0, and if above, will be 
				 set to 1.

				 If the threshold is set to 0, then an Otsu thresholding will
				 be returned.

	Outputs:
	- thresholded_image: a matrix representation of the thresholded image.
						 this is essentially a black and white image.
	- thresh: the threshold value

	To screen: the black-and-white image representation.
	- 
	"""
	if threshold == 0:
		thresh = threshold_otsu(image)

	if threshold != 0:
		thresh = threshold

	thresholded_image = closing(image > thresh, square(3), out=None)
	imshow(thresholded_image)

	return thresholded_image, thresh
	#return thresholded_image, thresh

def entropy_image(image, disk_size):
	"""
	This function takes in a single-channel image and returns it's entropy 
	within a disk size around it.

	Inputs:
	- image: a single-channel image matrix.
	- disk_size: a pixel radius to compute the entropy around.

	Outputs:
	- entropied: a matrix representation of an image

	To screen: a rainbow-colored image that represents the entropy.
	"""
	entropied = entropy(image, disk(disk_size))
	imshow(entropied, cmap = plt.cm.jet)
	return entropied

def find_overlap(image1, image2):
	"""
	This function takes in two images represented as matrices, and returns
	the overlapping regions between the two of them.

	Inputs:
	- image1, image2: two single-channel image matrices.

	Outputs:
	- overlap: matrix representation of the pixels turned "on" in both images

	To screen: a black-and-white representation of the overlap.
	"""
	overlap = np.logical_and(image1, image2)
	imshow(overlap)
	return overlap

def quantify_overlap(overlapping_image):
	"""
	This is syntactic sugar for quantifying the number of TRUE pixels 
	returned from the find_overlap function.
	"""
	return np.sum(overlapping_image)

def save_overlap(filename, overlapping_image):
	"""
	This is syntactic sugar for saving the overlapping_image derived from
	the quantify_overlap function.
	"""
	imsave(filename, overlapping_image)