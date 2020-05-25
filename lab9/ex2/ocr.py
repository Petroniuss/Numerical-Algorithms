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
def add_whitesymbols(matched, templates, space_size):
    """
        Adds white symbols,
        Turns matched matrix into text.
    """
    text = []
    h, w = len(matched), len(matched[0])
    for j in range(h):
        prev_match = None
        for i in range(w):
            if matched[j][i] is not None:
                if prev_match is not None:
                    y, x, prev_char = prev_match
                    prev_char_width = templates[prev_char].shape[1]

                    dist = i - x - prev_char_width
                    dist += 3
                    # Now based on this distance we're going to insert spaces
                    no_spaces = int(dist / (space_size))
                    # print(prev_char, matched[j][i], no_spaces, dist, )
                    text.extend(' ' * no_spaces)

                text.append(matched[j][i])
                prev_match = (j, i, matched[j][i])

        # In case we matched sth in line, we append newline character
        if prev_match is not None:
            text.append('\n')

    return "".join(text)


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


def is_empty(position, template_dims, shift, taken):
    shift_x, shift_y = shift
    row, col = position
    th, tw = template_dims

    return np.all(taken[row-th + shift_y:row,
                        col-tw + shift_x: col] == 0)


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

    templates = utils.load_characters(fontname)
    taken = np.zeros(img.shape, dtype=np.float64)
    matched = [[None] * w for i in range(h)]

    # FIXME -> This parameters might need adjustment!
    shift_x, shift_y = 5, 5
    shift = shift_x, shift_y
    for char, template in templates.items():
        th, tw = template.shape
        for row, col in match(img, template, threshold):
            if is_empty((row, col), template.shape, shift, taken):
                taken[row-th + shift_y:row,
                      col-tw + shift_x: col] = 0.5 + (np.random.random() / 2)
                matched[row - th][col - tw] = char

    if peek:
        utils.peek(taken)

    # Load templte of space to figure out space size!
    space_templ = character_img(' ', fontname)
    space_size = space_templ.shape[1]

    text = add_whitesymbols(matched, templates, space_size)

    return text


def test_fonts():
    print('Measuring accuracy on famous quote from Lord of The Rings')
    print('-' * 24)
    for font in utils.fonts.keys():
        img, actual_text = utils.convert_textfile('lotr', font)
        matched_text = ocr(img, font, peek=False)
        accuracy = utils.measure_correctness(matched_text, actual_text)
        print(f'Font: {font} --> ' + '{:.2f}'.format(accuracy))
        print('-' * 24)


# verdana and georgia work greate.. Times adds dots at the begginging for some reason
# verdna -/
# georiga -/
if __name__ == "__main__":
    test_fonts()
