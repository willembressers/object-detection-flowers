import cv2
import numpy
import base64

class Bouquet:

	image = None

	def __init__(self, source, flags=cv2.IMREAD_COLOR):
		# Split incoming stream on type and data
		content_type, data = source.split(',')

		# Decode the data
		decoded = base64.b64decode(data)

		# Parse bytes to array
		array = numpy.asarray(bytearray(decoded), dtype=numpy.uint8)

		# read the bytes array
		self.image = cv2.imdecode(array, flags=flags) 


	def get_dominant_colors(self):
		"""Return counts, colors

			Analyses the image, and returns the dominant colors with it's corresponding counts
		"""
		Z = self.image.reshape((-1,3))

		# convert to numpy.float32
		Z = numpy.float32(Z)

		# define criteria, number of clusters(K) and apply kmeans()
		K = 3
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

		# get the colors
		keys, counts = numpy.unique(label, return_counts=True)
		colors = ['rgb({red}, {green}, {blue})'.format(red=int(color[2]), green=int(color[1]), blue=int(color[0])) for color in center]
		
		return colors, counts, keys