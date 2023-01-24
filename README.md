# Segmentation demo v2.0.0

This is the version trying to implement long callbacks for large data download.

**Issue**.    
1. targeted\_callback does not work with Dash 2.   
2. dash.no\_update is not JSON serializable for flash caching (needed for Dash long callback). 

The first version of the image segmentation demo integrated with the model registry and the compute sevice manager.  
To run this demo, `docker-compose` the followings (in the order):  
-	mlex\_api: b384722  
-	mlex\_model\_registry: 65b803a     
-  mlex\_dash\_segmentation\_demo

Then build the images of the models using the command `make build_docker`. (Currently fully supports random forest, pyMSDtorch, and kmeans model) 

It supports asynchronous job submissions and results showing: choose which (completed) training results from the list to use for segmenting. Similarly, choose which (completed) segmenting (deploy) results to show.
 
