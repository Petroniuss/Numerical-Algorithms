import matplotlib.pyplot as plt
import numpy as np
from numpy import fft, rot90, multiply
from PIL import Image


img = np.asarray(Image.open("resources/school.jpg").convert("RGB"))

school_of_fish = img[:, :, :2]
school_of_fish = fft.fft2(school_of_fish, axes=(0, 1, 2))

print(school_of_fish.shape)

fish = np.asarray(Image.open("resources/fish1.png").convert("RGB"))
fish = fish[:, :, :22]

print(rot90(fish, 2).shape)
w, h, *tai = school_of_fish.shape
fish = (fft.fft2(fish, axes=(0, 1, 2), s=(w, h, 2)))
fish = np.rot90(fish, 2)

absolute_correlations = np.sum(
    abs(fft.ifft2(multiply(school_of_fish, fish))), axis=(2))
max_correlation = np.amax(absolute_correlations)

new_img = np.array(Image.open("resources/school.jpg").convert("RGB"))

for i in range(absolute_correlations.shape[0]):
    for j in range(absolute_correlations.shape[1]):
        if absolute_correlations[i, j] >= 0.1 * max_correlation:
            new_img[i, j][0] = 200

result = Image.fromarray(new_img)
result.save("new_school_of_fish.jpg")
