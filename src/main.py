from util.strokes_to_image import strokes_to_image
import json

with open("./pictionary_stroke_data/aircraft carrier.ndjson", "r") as file:
	line = file.readline()
	obj = json.loads(line)

	img = strokes_to_image(obj["drawing"])
	img.save("test.png", "PNG")
	



