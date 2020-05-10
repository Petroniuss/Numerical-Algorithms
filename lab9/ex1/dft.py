import matplotlib.pyplot as plt
import numpy as np
import cv2

from numpy.fft import fft2, ifft2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

resources_path = './resources/'
fish_repr_filename = resources_path + 'fish1.png'
school_filename = resources_path + 'school.jpg'

letter_filename = resources_path + 'galia_e.png'
galia_filename = resources_path + 'galia.png'


def imread_grayscale(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    return cv2.bitwise_not(img)


def imread_red_channel(filename):
    img = cv2.imread(filename)
    img = img[:, :, 2]

    return img


def surface_plot(values, **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    height, width = values.shape
    X = np.arange(width)
    Y = np.arange(height)
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, values, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, **kwargs)

    plt.show()


def match_pattern(img_filename, pattern_filename,
                  threshold=.9, color=(255, 0, 0), red_channel=False, box=True):
    # Load grayscale inverted images
    imreadF = imread_grayscale
    # Load red channel (for school)
    if red_channel:
        imreadF = imread_red_channel

    im = imreadF(img_filename)
    pattern = imreadF(pattern_filename)

    # Compute convolution
    correlation = ifft2(
        fft2(im, im.shape) * np.rot90(fft2(pattern, im.shape), 2), im.shape)
    # Complex -> Real
    correlation = np.abs(correlation)

    # Normalize
    maxx = np.amax(correlation)
    correlation = correlation / maxx

    # Box borders
    ptrn_h, ptrn_w = pattern.shape
    row_border = np.full(shape=(1, ptrn_w, 3), fill_value=color)
    col_border = np.full(shape=(1, ptrn_h, 3), fill_value=color)

    # Image with boxes around found pattern
    matched = cv2.imread(img_filename)
    matched = cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)

    height, width = im.shape
    found = 0
    for j in range(height):
        for i in range(width):
            if correlation[j, i] >= threshold:
                if not box:
                    if color[0] != 0:
                        matched[j, i][0] = color[0]
                    if color[1] != 0:
                        matched[j, i][1] = color[1]
                    if color[2] != 0:
                        matched[j, i][2] = color[2]
                else:
                    w, h = min(ptrn_w, width - i -
                               1), min(ptrn_h, height - j - 1)
                    # Top border
                    matched[j, i:i+w] = row_border[0, :w, :]
                    # Left border
                    matched[j:j+h, i] = col_border[0, :h]
                    # Bottom border
                    matched[j+h, i:i+w] = row_border[0, :w]
                    # Right border
                    matched[j:j+h, i+w] = col_border[0, :h]

                found += 1

    plt.imshow(matched)

    return found
