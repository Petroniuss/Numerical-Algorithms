import utils
import numpy as np
import cv2
from skimage.feature import peak_local_max
from numpy.fft import fft2, ifft2
from utils import *


def denoise(img):
    """
        Returns denoised <=> color-reversed binary image.
    """
    return to_binary(cv2.bitwise_not(img))


def cross_correlation(img, template):
    """
        Calculates cross-correlation between image and template.
    """
    return np.real(ifft2(fft2(img) * fft2(np.rot90(template, 2), img.shape)))


# FIXME
def align(img):
    """
        Aligns text present in input image, by calculating 2d rectangle encapsulating it.
        Note that the input should be binary image!
    """
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns values in range [-90,0) as the rectangle rotates clockwise angle goes to 0,
    # so we need to add 90 degrees to the angle
    # otherwise, we just take the inverse of the angle to make it positive
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # rotate and align the image
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = utils.to_binary(rotated)

    return rotated


# TODO
def add_whitesymbols(matched):
    """
        Adds white symbols
    """
    pass


def match(img, template, threshold):
    """
        Returns indicies (row, col) of found matches.
    """
    corr = cross_correlation(img, template)
    max_corr = np.max(corr)

    # corr /= max_corr
    # threshold = max(threshold, min(1.0, 3.8 * white_pixels_density(template)))

    threshold = threshold * max(max_corr, threshold * white_pixels(template))
    corr[corr < threshold] = 0

    return peak_local_max(corr, indices=True)


def ocr(img, fontname, threshold=.95, peek=False):
    """
       ->> Optical Character Recognition <<-
    """
    img = denoise(img)
    img = align(img)
    img = normalize(img)
    h, w = img.shape

    if peek:
        utils.peek(img)
    characters_dict = utils.load_characters(fontname)

    matched = [[None] * w for i in range(h)]

    templates = utils.load_characters(fontname)
    found_characters = np.zeros(img.shape, dtype=np.uint8)
    # FIXME -> This parameters might need adjustment!
    shift_x, shift_y = 5, 5
    counter = 0
    for char, template in templates.items():
        if counter == 0:
            print(char)
        ch, cw = template.shape
        for row, col in match(img, template, threshold):
            taken = False
            for y in range(row - ch + shift_y, row):
                for x in range(col - cw + shift_x, col):
                    if found_characters[y, x] != 0:
                        taken = True
                        break
                if taken:
                    break

            if not taken:
                found_characters[row-ch + shift_y:row,
                                 col-cw + shift_x: col] = ord(char)
                matched[row - ch][col - cw] = char
                counter += 1

    if peek:
        utils.peek(found_characters)

    text = ''
    for j in range(h):
        for i in range(w):
            if matched[j][i] is not None:
                text += matched[j][i]

    return text


# verdana and georgia work greate.. Times adds dots at the begginging for some reason
if __name__ == "__main__":
    img, actual_text = utils.convert_textfile('lotr', 'verdana')
    matched_text = ocr(img, 'verdana')
    print(actual_text)
    print(matched_text)
    print(utils.measure_correctness(matched_text, actual_text))

    pass
