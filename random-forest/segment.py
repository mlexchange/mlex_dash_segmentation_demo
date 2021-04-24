import numpy as np
import skimage
import argparse
from skimage import filters, feature
import pathlib
import imageio

import PIL.Image
from joblib import Parallel, delayed
import joblib
import feature_generation



"""
This script takes in as input an image filename and a jsonblob/list of features 
to generate. Each feature corresponds to a column in the outputed feature matrix, 
which is then saved to a file.
code from: https://github.com/plotly/dash-sample-apps/blob/d96997bd269deb4ff98b810d32694cc48a9cb93e/apps/dash-image-segmentation/trainable_segmentation.py#L64
"""

if __name__ == "__main__":
    ### GLOBAL VARIABLES ###
    parser = argparse.ArgumentParser()
    parser.add_argument('image_stack', help='image filepath')
    parser.add_argument('model_f', help='model filepath')
    parser.add_argument('output_dir', help='directory for output of classifier')
    args = parser.parse_args()
    MODEL_F = pathlib.Path(args.model_f)
    IMAGE_STACK = pathlib.Path(args.image_stack)
    OUTPUT_DIR = pathlib.Path(args.output_dir)

    # read in tiff stack to work on
    im_stack = imageio.volread(IMAGE_STACK)

    # read in model file
    clf = joblib.load(MODEL_F)
    def seg_image(im,index):
        features = feature_generation.multiscale_basic_features(
                im,
                multichannel=False,
                intensity=True,
                edges=False,
                texture=False,
                )
        features = features.reshape(features.shape[0], features.shape[1]*features.shape[2]).T
        output = clf.predict(features)
        output_im = output.reshape(im.shape) #reassemble list of masked pixels into an image
        output_f_name = OUTPUT_DIR / '{}-.dat'.format(index)
        np.savetxt(str(output_f_name), output_im)
        imageio.imsave(str(OUTPUT_DIR / '{}-classified.tif'.format(index)), output_im) 
        print('classified: {}'.format(index))

    Parallel(n_jobs=-1)(delayed(seg_image)(im_stack[i], i) for i in range(100))


