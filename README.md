# Segmentation demo v2.0.0

The first version of the image segmentation demo integrated with the model registry and the compute sevice manager. It is adapted from a [segmentation example](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-image-segmentation) in [Dash Enterprise App Gallery](https://dash.gallery/dash-image-segmentation/).

To run this demo, `docker-compose` the followings (in the order):  
-	mlex\_api: b384722  
-	mlex\_model\_registry: 65b803a     
-  mlex\_dash\_segmentation\_demo

Then build the images of the models using the command `make build_docker`. (Currently fully supports random forest, pyMSDtorch, and kmeans model) 

It supports asynchronous job submissions and results showing: choose which (completed) training results from the list to use for segmenting. Similarly, choose which (completed) segmenting (deploy) results to show.

# Copyright
MLExchange Copyright (c) 2021, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
 
