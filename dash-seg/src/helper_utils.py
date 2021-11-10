import imageio
import matplotlib.pyplot as plt
import json
import numpy as np
import io
import base64
import PIL.Image
import plotly.express as px
import urllib
import requests

class_label_colormap = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]

#### HELPER UTILS
def dcm_to_np(dir_path):
    np_volume = imageio.volread(dir_path)
    # np_volume = imageio.volread("imageio:stent.npz")
    return np_volume


def class_to_color(n):
    return class_label_colormap[n]


def color_to_class(c):
    return class_label_colormap.index(c)


def shapes_to_key(shapes):
    return json.dumps(shapes)


def label_to_colors(img, colormap=px.colors.qualitative.Light24, alpha=128, color_class_offset=0):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """

    def fromhex(n):
        return int(n, base=16)

    colormap = [
        tuple([fromhex(h[s: s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]
    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)
    for c in range(int(minc), int(maxc) + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]
    return np.concatenate(
        (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
    )


def store_shapes_seg_pair(d, key, seg, remove_old=True):
    """
    Stores shapes and segmentation pair in dict d
    seg is a PIL.Image object
    if remove_old True, deletes all the old keys and values.
    """
    bytes_to_encode = io.BytesIO()
    seg.save(bytes_to_encode, format="png")
    bytes_to_encode.seek(0)
    data = base64.b64encode(bytes_to_encode.read()).decode()
    if remove_old:
        return {key: data}
    d[key] = data
    return d


def look_up_seg(d, key):
    """ Returns a PIL.Image object """
    data = d[key]
    img_bytes = base64.b64decode(data)
    img = PIL.Image.open(io.BytesIO(img_bytes))
    return img


def make_default_figure(image_index, np_volume, shapes=[], stroke_color='#ff4f00', image_cache=None):
    if image_cache is None:
        im = np_volume[image_index]
        width, height = im.shape[0:2]
        print('default figure')
    else:
        # image_cache is a dict, keys=filename, value=bin encoding
        img_bytes = base64.b64decode(image_cache)
        im_bbytes = io.BytesIO(img_bytes)
        print('first', im_bbytes)
        im = PIL.Image.open(im_bbytes)
        print('read in uploaded file {}'.format(im))
        width, height = np.array(im).shape[0:2]
        print(width, height)

    fig = px.imshow(im, binary_string=True)
    fig.update_layout(
        {
            'dragmode': 'drawopenpath',
            'shapes': shapes,
            'newshape.line.color': stroke_color,
            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
        }
    )
    fig.update_xaxes(
        showgrid=False, range=(0, width), showticklabels=False, zeroline=False
    )
    fig.update_yaxes(
        showgrid=False,
        scaleanchor="x",
        range=(height, 0),
        showticklabels=False,
        zeroline=False,
    )
    return fig
    
    
def get_job(user, mlex_app, job_type=None, deploy_location=None):
    url = 'http://job-service:8080/api/v0/jobs?'
    #url = 'http://host.docker.internal:8080/api/v0/jobs?'
    if user:
        url += ('&user='+user)
    if mlex_app:
        url += ('&mlex_app='+mlex_app)
    if job_type:
        url += ('&job_type='+job_type)
    if deploy_location:
        url += ('deploy_location'+deploy_location)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


def post_job(job):
    url = 'http://job-service:8080/api/v0/jobs'
    #url = 'http://host.docker.internal:8080/api/v0/jobs'
    job_dict = {"user": job.user,
                "mlex_app": job.mlex_app,
                "job_type": job.job_type,
                "description": job.description,
                "deploy_location": job.deploy_location,
                "gpu": job.gpu,
                "data_uri": job.data_uri,
                "container_uri": job.container_uri,
                "container_cmd": job.container_cmd,
                "container_kwargs": job.container_kwargs
               }
    print(f'job dict\n{json.dumps(job_dict)}')
    return requests.post(url, json=job_dict).status_code
   
   
   
   