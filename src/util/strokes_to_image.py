from PIL import Image, ImageDraw

def strokes_to_image(strokes, image_size=256, line_width=5):
    image = Image.new('L', (image_size, image_size), 255)
    draw = ImageDraw.Draw(image)

    for stroke in strokes:
        x, y = stroke[0], stroke[1]
        for i in range(1, len(x)):
            draw.line([(x[i-1], y[i-1]), (x[i], y[i])], fill=0, width=line_width)
            
    return image