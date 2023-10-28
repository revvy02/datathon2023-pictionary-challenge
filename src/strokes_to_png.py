from PIL import Image, ImageDraw

def strokes_to_png(strokes, image_size=256, line_width=5, save_path=None):
    """
    Convert stroke data to PNG.
    
    Parameters:
    - strokes: List of strokes, where each stroke is a list with two lists for X and Y coordinates.
    - image_size: Size of the output image (default is 256x256).
    - line_width: Thickness of the lines in the image.
    - save_path: Path to save the image. If None, the image is returned without saving.
    
    Returns:
    - PIL.Image object if no save path is provided. None otherwise.
    """
    # Create a white background image
    image = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(image)
4
    for stroke in strokes:
        x, y = stroke[0], stroke[1]
        for i in range(1, len(x)):
            draw.line([x[i-1], y[i-1], x[i], y[i]], fill='black', width=line_width)
            
    return image
