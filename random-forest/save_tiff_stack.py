import numpy
import imageio
import matplotlib.pyplot as plt

ims = imageio.mimread('./sample-data/bead_pack.tif')

for i,im in enumerate(ims):
    imageio.imsave('sample-data/images/{}_bead_pack.tif'.format(i), im)
