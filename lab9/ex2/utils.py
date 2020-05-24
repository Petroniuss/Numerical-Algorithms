from PIL import ImageFont, ImageDraw, Image, ImageOps
import numpy as np
import pylcs
import string
import cv2

# One font size is assumed!
FONT_SIZE = 32
CHARACTERS = string.ascii_lowercase + \
    string.ascii_uppercase + string.digits + "./,?!"

fonts_dir = './resources/fonts/'
img_dir = './resources/imgs/'
text_dir = './resources/texts/'

fonts = {
    'georgia': fonts_dir + 'georgia.ttf',
    'times': fonts_dir + 'times-new-roman.ttf',
    'verdana': fonts_dir + 'verdana.ttf'
}


def convert_textfile(text_filename, fontname, output_filename=None):
    """
        Wrapper over `convert`.
        Returns tuple: (img, text).
    """
    text = None
    with open(text_dir + text_filename + '.txt', 'r') as file:
        text = file.read()

    if output_filename is None:
        output_filename = text_filename

    return convert(text, fontname, output_filename), text


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


def character_img(chr, fontname):
    """
        Returns image of given character as numpy array.
    """
    font = ImageFont.truetype(fonts[fontname], size=FONT_SIZE)
    w, h = font.getsize(chr)

    M = np.zeros((h, w), dtype=np.uint8)

    img = Image.fromarray(M)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), chr, font=font, fill=(255))

    img = ImageOps.invert(img)

    return np.array(img)


def load_characters(fontname):
    """
        Loads images of characters into dictionary.
        Note that they're already color-reversed and converted to binary images!
    """
    imgs = {}
    for chr in CHARACTERS:
        imgs[chr] = cv2.bitwise_not(to_binary(character_img(chr, fontname)))

    return imgs


def measure_correctness(matched_text, actual_text):
    n = len(actual_text)
    lcs = pylcs.lcs(matched_text, actual_text)

    return lcs / n * 100.0


def to_binary(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def peek(img):
    cv2.startWindowThread()
    cv2.imshow("Optical Character Recogintion", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_size(img):
    w, h = img.shape
    return w * h


def white_pixels(img):
    return len(np.argwhere(img == 255))


def white_pixels_density(img):
    """
        Returns ration between white pixels and whole image.
    """
    return white_pixels(img) / img_size(img)
