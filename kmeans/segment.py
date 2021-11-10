import os
import numpy as np
import pathlib
import argparse
import imageio
from joblib import load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_stack', help='image filepath')
    parser.add_argument('model_file', help='model filepath')
    parser.add_argument('output_dir', help='directory for outputs')
    args = parser.parse_args()
    image_stack = pathlib.Path(args.image_stack)
    model_file = pathlib.Path(args.model_file)
    output_dir = pathlib.Path(args.output_dir)
    
    images = imageio.volread(image_stack)
    shp = images.shape
    test_images = images.reshape((shp[0]*shp[1]*shp[2], 1))
    
    kmeans = load(model_file)
    outputs = kmeans.predict(test_images).reshape(shp[0], shp[1], shp[2])

    for index, out in enumerate(outputs):
        pth = os.path.join(args.output_dir, f'{index}')
        np.savetxt(pth, out)
        imageio.imsave(pth+'test.tif', out)
