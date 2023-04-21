import io
import json
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import base64
from copy import deepcopy
import imageio
import numpy as np
import pandas as pd
import PIL.Image
import plotly.express as px
import plotly.graph_objects as go
import requests
import urllib

class_label_colormap = ["#E69F00", "#8F00FF", "#56B4E9", "#009E73", "#F0E442", "#ff4f00"]
RGB_colormap = [(230,159,0), (143,0,255), (86,180,233), (0,158,115), (240,228,66), (255,79,0)]


#### HELPER UTILS
def remove_key_from_dict_list(data, key):
    new_data = []
    for item in data:
        if key in item:
            new_item = deepcopy(item)
            new_item.pop(key)
            new_data.append(new_item)
        else:
            new_data.append(item)
    
    return new_data

def dcm_to_np(dir_path):
    np_volume = imageio.volread(dir_path)
    # np_volume = imageio.volread("imageio:stent.npz")
    if len(list(np_volume.shape)) == 2:
        np_volume = np_volume.reshape(1,np_volume.shape[0],np_volume.shape[1])
    return np_volume


def class_to_color(n):
    return class_label_colormap[n]


def color_to_class(c):
    return class_label_colormap.index(c)


def shapes_to_key(shapes):
    return json.dumps(shapes)


def label_to_colors(img, colormap=RGB_colormap, alpha=128, color_class_offset=0):
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

#     colormap = [
#         tuple([fromhex(h[s: s + 2]) for s in range(0, len(h), 2)])
#         for h in [c.replace("#", "") for c in colormap]
#     ]
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


def make_default_figure(image_index, np_volume, shapes=[], stroke_color=class_label_colormap[-1], stroke_width=3, image_cache=None):
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
            'newshape.line.width': stroke_width,
            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}
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


def generate_figure(log, start):
    end = log.find('number of network parameters')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n','\n')), sep='\t')
    try:
        fig = px.line(df)
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss", margin=dict(l=20, r=20, t=20, b=20))
        return fig
    except Exception:
        return go.Figure(go.Scatter(x=[], y=[]))


def get_job(user, mlex_app):
    url = 'http://job-service:8080/api/v0/jobs?'
    if user:
        url += ('&user=' + user)
    if mlex_app:
        url += ('&mlex_app=' + mlex_app)
    
    response = requests.get(url).json()
    return response


def init_counters(user, job_type):
    job_list = get_job(user, 'seg-demo')
    if job_list is not None:
        for job in reversed(job_list):
            last_job = job['job_kwargs']['kwargs']['job_type'].split()
            value = int(last_job[-1])
            last_job = ' '.join(last_job[0:-1])
            if last_job == job_type:
                return value + 1
    return 0


