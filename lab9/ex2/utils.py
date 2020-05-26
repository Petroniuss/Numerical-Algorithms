from PIL import ImageFont, ImageDraw, Image, ImageOps
from collections import Counter
import PIL
import numpy as np
import string
import cv2

# One font size is assumed!
FONT_SIZE = 32
CHARACTERS = string.ascii_lowercase + \
    string.ascii_uppercase + string.digits + "./,?!-"

fonts_dir = './resources/fonts/'
img_dir = './resources/imgs/'
text_dir = './resources/texts/'

fonts = {
    'georgia': fonts_dir + 'georgia.ttf',  # serif
    'firasans': fonts_dir + 'firasans.ttf',  # sans-serif
    'verdana': fonts_dir + 'verdana.ttf'  # sans-serif
}


def convert_textfile(text_filename, fontname, output_filename=None, angle=0):
    """
        Wrapper over `convert`.
        Returns tuple: (img, text).
    """
    text = None
    with open(text_dir + text_filename + '.txt', 'r') as file:
        text = file.read()

    if output_filename is None:
        output_filename = text_filename

    return convert(text, fontname, output_filename, angle), text


def convert(text, fontname, output_filename=None, angle=0):
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

    img = img.rotate(-angle, resample=PIL.Image.BILINEAR, expand=1)
    img = ImageOps.invert(img)
    if output_filename is not None:
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
        Loads images of characters into dictionary,
        and returns sorted dictionary based on amount of white pixels.

        Note that they're already color-reversed and converted to binary images!

        The idea here is that larger templates (containing more white pixels)
        might have smaller templates inside of them for example e and c or T and I.
        To avoid mismatch we first check bigger ones!

    """
    templates = {}
    for chr in CHARACTERS:
        template = cv2.bitwise_not(
            to_binary(character_img(chr, fontname)))
        templates[chr] = normalize(template)

    def sort_by(item): return -white_pixels(item[1])
    # def sort_by(item): return -white_pixels_density(item[1])

    templates = {k: v for (k, v) in sorted(
        templates.items(), key=sort_by)}

    return templates


def measure_correctness_lcs(matched_text, actual_text):
    """
       Measures correctness by calculating longest common substring. 
    """
    n = len(actual_text)
    lcs_len = lcs(matched_text, actual_text)

    return lcs_len / n * 100.0


def calc_occurences(matched_text):
    """
        Calcualtes occurences of each character.
    """
    m_co = Counter(matched_text)

    return m_co


def to_binary(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def normalize(img):
    return img / 255


def peek(img):
    cv2.startWindowThread()
    cv2.imshow("Optical Character Recogintion", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_size(img):
    w, h = img.shape
    return w * h


def white_pixels(img):
    return np.count_nonzero(img == 1)


def white_pixels_density(img):
    """
        Returns ration between white pixels and whole image.
    """
    return white_pixels(img) / img_size(img)


def lcs_table(x, y):
    """
        Returns table construced by dynamic algorithm for finding lcs.
    """
    m, n = len(x), len(y)
    dp = np.empty((m + 1, n + 1))

    dp[0] = np.zeros(n + 1)
    dp[:, 0] = np.zeros(m + 1)

    for j in range(1, m + 1):
        for i in range(1, n + 1):
            if x[j - 1] == y[i - 1]:
                dp[j, i] = 1 + dp[j - 1, i - 1]
            else:
                dp[j, i] = max(dp[j - 1, i], dp[j, i - 1])

    return dp


def lcs(x, y):
    """
        Returns len of longest common subsequence for given strings
    """
    dp = lcs_table(x, y)
    seq = []
    ptrs = [[-1, 0],
            [0, -1]]
    xs = np.empty(2)
    j, i = len(x), len(y)
    while j != 0 and i != 0:
        if x[j - 1] == y[i - 1]:
            seq.append(x[j - 1])
            j, i = j - 1, i - 1
        else:
            for k in range(2):
                xs[k] = dp[j + ptrs[k][1], i + ptrs[k][0]]
            k = np.argmax(xs)
            j, i = j + ptrs[k][1], i + ptrs[k][0]

    return len(seq)
