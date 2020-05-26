import utils
import numpy as np
import cv2
import sys
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


def align(img):
    """
        Aligns text present in input image, by calculating 2d rectangle encapsulating it.
        Note that the input should be binary image!
    """
    coords = np.column_stack(np.where(img > 0))

    neg_angle = cv2.minAreaRect(coords)[-1]
    angle = -neg_angle
    if angle < -45:
        angle -= 90

    h, w = img.shape
    rot_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE, borderValue=0)
    rotated = utils.to_binary(rotated)

    return rotated


def add_whitesymbols(matched, templates, space_size):
    """
        Adds white symbols,
        Turns matched matrix into text.
    """
    text = []
    h, w = len(matched), len(matched[0])
    j, i = 0, 0
    while j < h:
        prev_match = None
        line_matches = 0
        i = 0
        while i < w:
            if matched[j][i] is not None:
                if prev_match is not None:
                    y, x, prev_char = prev_match
                    prev_char_width = templates[prev_char].shape[1]

                    # Now based on this distance we're going to insert spaces
                    dist = i - x - prev_char_width
                    dist += 3
                    no_spaces = int(dist / (space_size))
                    text.extend(' ' * no_spaces)

                text.append(matched[j][i])
                prev_match = (j, i, matched[j][i])
                i += templates[matched[j][i]].shape[1] // 2
                line_matches += 1
            else:
                i += 1

        # In case we matched sth in line, we append newline character
        # Somewhat ugly hack for dealing with dots ;)
        if prev_match is not None:
            flag = True
            for fi in range(line_matches):
                if text[-fi - 1] not in set(['.', ',']):
                    flag = False
                    break
            if flag:
                for fi in range(line_matches):
                    text.pop()
                j += 1
            else:
                text.append('\n')
                j += templates[prev_match[2]].shape[0] // 2
        else:
            j += 1

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

       Arguments:
            - img -> numpy 2d array representing img in grayscale (0 - 255)
            - fontname -> one of supported fonts (check utils.py)
            - threshold -> this one might mess up whole thing ;)
            - peek -> whether to show partial results in a popup window.
        Returns:
            - matched text
            - image of boundaries of matched characters
    """
    if peek:
        utils.peek(img)

    img = denoise(img)
    img = align(img)
    img = normalize(img)
    h, w = img.shape
    if peek:
        utils.peek(img)

    templates = utils.load_characters(fontname)
    taken = np.zeros(img.shape, dtype=np.float64)
    matched = [[None] * w for i in range(h)]

    # Values yielding good results shift_x: [2..5]
    shift_x, shift_y = 3, 5
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

    return text, taken


def ocr_textfile(text_filename, fontname, angle=0, output_filename=None, peek=False):
    """
        Wrapper over ocr which takes textfile. 
        Note that the file should be in texts directory!
    """
    img, actual_text = utils.convert_textfile(
        text_filename, fontname, output_filename, angle)
    matched_text, taken = ocr(img.copy(), fontname, peek=peek)
    accuracy = utils.measure_correctness_lcs(matched_text, actual_text)

    return img, taken, matched_text, accuracy


def ocr_text(text, fontname, angle=0, output_filename=None, peek=False):
    """
        Wrapper over ocr which takes plain text.
    """

    img = utils.convert(text, fontname, output_filename, angle)
    matched_text, taken = ocr(img.copy(), fontname, peek=peek)
    accuracy = utils.measure_correctness_lcs(matched_text, text)

    return img, taken, matched_text, accuracy


def run_with_stats(text_filename, fontname, angle, peek):
    img, _, matched_text, accuracy = ocr_textfile(
        text_filename, fontname, angle=angle, peek=peek)
    print(f'Font: {fontname} --> ' +
          '{:.2f}%'.format(accuracy), end='\n\n')
    print(matched_text)
    print('-' * 32)


if __name__ == "__main__":
    """
        To run from command line pass 
            0 - args, test all fonts on a preapred file.
            2+ - 
                1 - textfile located in resources/texts/ (pass a name, omit .txt)
                2 - fontname (check utils.py)
                3 - angle (small angles are supported: [-45, 45])
                4 - peek (boolean) indicating whether to show images in a popup 

            Example call:
                python ocr.py lotr verdana 15 True
    """
    args_num = len(sys.argv)
    if args_num < 3:
        print('Measuring accuracy on famous quote from Lord of The Rings')
        print('-' * 64)
        for fontname in utils.fonts.keys():
            run_with_stats('lotr', fontname, 0, False)
    else:
        text_filename = sys.argv[1]
        fontname = sys.argv[2]
        angle = 0
        if args_num > 3:
            angle = int(sys.argv[3])
        peek = False
        if args_num > 4:
            peek = bool(sys.argv[4])

        print('   ->> Optical Character Recognition <<- ')
        print('-' * 64)
        run_with_stats(text_filename, fontname, angle, peek)
