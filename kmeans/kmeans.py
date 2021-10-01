import os
import json
import argparse
import pathlib
import imageio
from joblib import dump
from sklearn.cluster import KMeans
from model_validation import TrainingParameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_stack', help='image filepath')
    parser.add_argument('model_dir', help='path to model (output) directory')
    parser.add_argument('parameters', help='dictionary that contains training parameters')
    args = parser.parse_args()

    images = imageio.volread(pathlib.Path(args.image_stack))
    shp = images.shape
    training_images = images.reshape((shp[0]*shp[1]*shp[2],1))

    # parse parameters
    if args.parameters is not None:
        parameters = TrainingParameters(**json.loads(args.parameters))
    
    kmeans = KMeans(n_clusters=parameters.n_clusters,
                    init=parameters.init,
                    n_init=parameters.n_init,
                    max_iter=parameters.max_iter,
                    tol=parameters.tol,
                    random_state=parameters.random_state,
                    algorithm=parameters.algorithm)

    kmeans.fit(training_images)

    # Save model
    pth = os.path.join(args.model_dir, 'kmeans.joblib')
    dump(kmeans, pth)

    print('trained k-means: {}'.format(pth))
