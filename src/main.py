from util.strokes_to_image import strokes_to_image
from util.image_to_array import image_to_array
from util.array_to_image import array_to_image

import json
import numpy as np

with open("./pictionary_stroke_data/aircraft carrier.ndjson", "r") as file:
	line = file.readline()
	obj = json.loads(line)

	img = strokes_to_image(obj["drawing"])
	arr = image_to_array(img)

	re_img = array_to_image(arr)
	re_img.save("test.png", "PNG")
	



