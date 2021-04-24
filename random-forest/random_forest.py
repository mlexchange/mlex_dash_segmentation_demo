from sklearn.ensemble import RandomForestClassifier
import argparse
import pathlib
import glob
import skimage
import numpy as np
import PIL.Image
import joblib

""" Train a random forest classifier
    Input: design matrix, labelled masks
    Output: trained model, its-a-pickle!

code from: https://github.com/plotly/dash-sample-apps/blob/d96997bd269deb4ff98b810d32694cc48a9cb93e/apps/dash-image-segmentation/trainable_segmentation.py#L130
"""
def img_to_ubyte_array(img):
    """
    PIL.Image.open is used so that a io.BytesIO object containing the image data
    can be passed as img and parsed into an image. Passing a path to an image
    for img will also work.
    """
    ret_ = skimage.util.img_as_ubyte(np.array(PIL.Image.open(img)))
    return ret_


def fit_segmenter(labels, features, clf):
    """
    Segmentation using labeled parts of the image and a classifier.
    Parameters
    ----------
    labels : ndarray of ints
        Image of labels. Labels >= 1 correspond to the training set and
        label 0 to unlabeled pixels to be segmented.
    features : ndarray
        Array of features, with the first dimension corresponding to the number
        of features, and the other dimensions correspond to ``labels.shape``.
    clf : classifier object
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    Returns
    -------
    output : ndarray
        Labeled array, built from the prediction of the classifier trained on
        ``labels``.
    clf : classifier object
        classifier trained on ``labels``
    Raises
    ------
    NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
    """
    training_data = features[:, labels > 0].T
    training_labels = labels[labels > 0].ravel()
    clf.fit(training_data, training_labels)
    data = features[:, labels == 0].T
    predicted_labels = clf.predict(data)
    output = np.copy(labels)
    output[labels == 0] = predicted_labels
    return output, clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_dir', help='path to mask directory')
    parser.add_argument('feature_dir', help = 'path to feature directory')
    parser.add_argument('model_dir', help = 'path to model (output) directory')

    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    ### READ IN FEATURE LIST ###
    feature_glob = pathlib.Path(args.feature_dir)

    f_list = [np.genfromtxt(f) for f in feature_glob.glob("*feature")]
    all_features =np.concatenate(f_list, axis=-1).T

    ### READ IN IMAGE LIST ###
    mask_glob = pathlib.Path(args.mask_dir)
    mask_list = [np.genfromtxt(im).ravel() for im in mask_glob.glob('n-*')]
    all_mask = np.concatenate(mask_list)


    ### CHECK THAT n_features == n_images
    assert(len(all_features) == len(all_mask))
    train_features = all_features[ all_mask>0,:]
    train_mask = all_mask[all_mask >0]

    ### CREATE RANDOM FOREST CLF ###
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=8, max_samples=0.05)

    clf.fit(train_features,train_mask)
    model_output_name = model_dir / 'random-forest.model'
    joblib.dump(clf, model_output_name)
    print('trained random forest: {}'.format(model_output_name))
    

