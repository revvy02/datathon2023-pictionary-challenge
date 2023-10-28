import json
import numpy as np

from Processor import Processor

def main():
	# init model
	with open("./pictionary_stroke_data/aircraft carrier.ndjson", "r") as file:
		line = file.readline()
		obj = json.loads(line)
		p = Processor(image_size=256, line_width=5)

		img = p.strokes_to_image(obj["drawing"])
		arr = p.image_to_array(img)
		
		re_img = p.array_to_image(arr)
		re_img.save("test.png", "PNG")


	# load data

	# train model with processed data

	# 

	





