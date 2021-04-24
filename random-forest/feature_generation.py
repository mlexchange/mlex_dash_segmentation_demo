import numpy as np
import skimage
import argparse
from skimage import filters, feature
import pathlib

import PIL.Image
from joblib import Parallel, delayed



"""
This script takes in as input an image filename and a jsonblob/list of features 
to generate. Each feature corresponds to a column in the outputed feature matrix, 
which is then saved to a file.
code from: https://github.com/plotly/dash-sample-apps/blob/d96997bd269deb4ff98b810d32694cc48a9cb93e/apps/dash-image-segmentation/trainable_segmentation.py#L64
"""

def img_to_ubyte_array(img):
    """
    PIL.Image.open is used so that a io.BytesIO object containing the image data
    can be passed as img and parsed into an image. Passing a path to an image
    for img will also work.
    """
    ret_ = skimage.util.img_as_ubyte(np.array(PIL.Image.open(img)))
    return ret_

def _texture_filter(gaussian_filtered):
    H_elems = [
        np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(range(gaussian_filtered.ndim), 2)
    ]
    eigvals = feature.hessian_matrix_eigvals(H_elems)
    return eigvals


def _mutiscale_basic_features_singlechannel(
    img, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16
):
    """Features for a single channel nd image.
    Parameters
    ----------
    """
    # computations are faster as float32
    img = np.ascontiguousarray(skimage.img_as_float32(img))
    sigmas = np.logspace(
        np.log2(sigma_min),
        np.log2(sigma_max),
        num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
        base=2,
        endpoint=True,
    )
    all_filtered = Parallel(n_jobs=-1, prefer="threads")(
        delayed(filters.gaussian)(img, sigma) for sigma in sigmas
    )
    features = []
    if intensity:
        features += all_filtered
    if edges:
        all_edges = Parallel(n_jobs=-1, prefer="threads")(
            delayed(filters.sobel)(filtered_img) for filtered_img in all_filtered
        )
        features += all_edges
    if texture:
        all_texture = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_texture_filter)(filtered_img) for filtered_img in all_filtered
        )
        features += itertools.chain.from_iterable(all_texture)
    return features

def multiscale_basic_features(
    image,
    multichannel=True,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16,
):
    """Local features for a single- or multi-channel nd image.
    Intensity, gradient intensity and local structure are computed at
    different scales thanks to Gaussian blurring.
    Parameters
    ----------
    image : ndarray
        Input image, which can be grayscale or multichannel.
    multichannel : bool, default False
        True if the last dimension corresponds to color channels.
    intensity : bool, default True
        If True, pixel intensities averaged over the different scales
        are added to the feature set.
    edges : bool, default True
        If True, intensities of local gradients averaged over the different
        scales are added to the feature set.
    texture : bool, default True
        If True, eigenvalues of the Hessian matrix after Gaussian blurring
        at different scales are added to the feature set.
    sigma_min : float, optional
        Smallest value of the Gaussian kernel used to average local
        neighbourhoods before extracting features.
    sigma_max : float, optional
        Largest value of the Gaussian kernel used to average local
        neighbourhoods before extracting features.
    Returns
    -------
    features : np.ndarray
        Array of shape ``(n_features,) + image.shape``
    """
    if image.ndim >= 3 and multichannel:
        all_results = (
            _mutiscale_basic_features_singlechannel(
                image[..., dim],
                intensity=intensity,
                edges=edges,
                texture=texture,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            for dim in range(image.shape[-1])
        )
        features = list(itertools.chain.from_iterable(all_results))
    else:
        features = _mutiscale_basic_features_singlechannel(
            image,
            intensity=intensity,
            edges=edges,
            texture=texture,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    return np.array(features, dtype=np.float32)

if __name__ == '__main__':
    # get dir, and feature extract all images

    ### GLOBAL VARIABLES ###
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help='image filepath')
    parser.add_argument('feature_dir', help='output filepath')
    args = parser.parse_args()
    OUTPUT_FEATURE_DIR = pathlib.Path(args.feature_dir)
    images_path = pathlib.Path(args.image_dir)

    ###INPUT_ARGS_HARDCORE
    feature_list = {'intensity': True,
                    'edges': False,
                    'texture': False}

    for im in images_path.glob('*.tif'):
        im_name_root = im.name.strip(im.suffix)
        image = img_to_ubyte_array(im)
        features = multiscale_basic_features(
                image,
                multichannel=False,
                intensity=feature_list['intensity'],
                edges=feature_list['edges'],
                texture=feature_list['texture']
                )
        num_features = features.shape[0]
        feature_out_name = str(OUTPUT_FEATURE_DIR / im_name_root)+'.feature'
        np.savetxt(feature_out_name, features.reshape(num_features,-1))
        print('features generated for: {}'.format(feature_out_name))
