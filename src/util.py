import json
import math
import os

import numpy as np

from PIL import Image, ImageDraw
from config import IMAGE_SIZE, LINE_WIDTH, TRAINING_SET_SIZE, VALIDATION_SET_SIZE, RAW_DATA_SOURCE

def strokes_to_image(strokes, image_size=IMAGE_SIZE, line_width=LINE_WIDTH):
	image = Image.new('L', (image_size, image_size), 255)
	draw = ImageDraw.Draw(image)

	for stroke in strokes:
		x, y = stroke[0], stroke[1]
		for i in range(1, len(x)):
			draw.line([(x[i-1], y[i-1]), (x[i], y[i])], fill=0, width=line_width)
			
	return image

def image_to_array(img, image_size=IMAGE_SIZE):
	img = img.resize((image_size, image_size))
	img = img.convert("L")
    
	img_arr = np.array(img)
	img_arr = img_arr / 255.0
	img_arr = np.expand_dims(img_arr, axis=0)
	
	return img_arr

def array_to_image(arr, image_size=IMAGE_SIZE):
	arr = arr.reshape(image_size, image_size)
	
	arr[arr == 1] = 255
	arr[arr == 0] = 0

	return Image.fromarray(arr.astype('uint8'), mode="L")

def create_data_object(line):
	input_obj = json.loads(line)
	output_obj = {}

	output_obj["image"] = image_to_array(strokes_to_image(input_obj["drawing"], IMAGE_SIZE, LINE_WIDTH), IMAGE_SIZE).tolist()
	output_obj["strokes"] = input_obj["drawing"]

	return output_obj

def get_training_set(file_name):
	with open(os.path.join(RAW_DATA_SOURCE, file_name), 'r') as file:
		lines = [file.readline() for _ in range(TRAINING_SET_SIZE)]

		return [json.dumps(create_data_object(line)) + "\n" for line in lines]
	
def get_validation_set(file_name):
	with open(os.path.join(RAW_DATA_SOURCE, file_name), 'r') as file:
		for _ in range(TRAINING_SET_SIZE):
			file.readline()

		lines = [file.readline() for _ in range(VALIDATION_SET_SIZE)]
		
		return [json.dumps(create_data_object(line)) + "\n" for line in lines]
	
