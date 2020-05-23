from PIL import ImageFont, ImageDraw, Image, ImageOps
import numpy as np
import cv2

# One font size is assumed!
FONT_SIZE = 24

fonts_dir = './resources/fonts/'
img_dir = './resources/imgs/'
text_dir = './resources/texts/'

fonts = {
    'georgia': fonts_dir + 'georgia.ttf',
    'times-new-roman': fonts_dir + 'times-new-roman.ttf',
}


def convert_textfile(text_filename, fontname, output_filename=None):
    """
        Wrapper over `convert`.
    """
    text = None
    with open(text_dir + text_filename, 'r') as file:
        text = file.read()

    if output_filename is None:
        output_filename = text_filename.split('.')[0]

    return convert(text, fontname, output_filename)


def convert(text, fontname, output_filename):
    """
        Converts given text to image using given fontname. 
        Saves output and returns image as numpy array in grayscale.
    """
    font = ImageFont.truetype(fonts[fontname], size=FONT_SIZE)
    w, h = 0, 0
    lines = text.split('\n')
    for line in lines:
        lineW, lineH = font.getsize(line)
        w = max(w, lineW)
        h += lineH

    off_x, off_y = 25, 40
    M = np.zeros((h + (2 * off_y), w + (2 * off_x)), dtype=np.uint8)

    img = Image.fromarray(M)
    draw = ImageDraw.Draw(img)
    draw.text((off_x, off_y), text, font=font, fill=(255))

    img = ImageOps.invert(img)
    img.save(img_dir + output_filename + '.png')

    return np.array(img)
