import numpy as np

from PIL import Image, ImageDraw

class Processor:
	def __init__(self, image_size=256, line_width=5):
		self._image_size = image_size
		self._line_width = line_width
	
	def strokes_to_image(self, strokes):
		image = Image.new('L', (self._image_size, self._image_size), 255)
		draw = ImageDraw.Draw(image)

		for stroke in strokes:
			x, y = stroke[0], stroke[1]
			for i in range(1, len(x)):
				draw.line([(x[i-1], y[i-1]), (x[i], y[i])], fill=0, width=self._line_width)
				
		return image
	
	def image_to_array(self, img):
		img = img.resize((self._image_size, self._image_size))
		img = img.convert("L")
		
		img_array = np.array(img)
		
		img_array[img_array == 255] = 1
		img_array[img_array == 0] = 0

		img_vector = img_array.flatten()
		
		return img_vector
	
	def array_to_image(self, arr):
		arr = arr.reshape(self._image_size, self._image_size)
		
		arr[arr == 1] = 255
		arr[arr == 0] = 0

		return Image.fromarray(arr.astype('uint8'), mode="L") 