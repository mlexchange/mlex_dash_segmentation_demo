# Segmentation demo v2.0.0

The first version of the image segmentation demo integrated with the model registry and the compute sevice manager.  
To run this demo, `docker-compose` the followings (in the order):  
-	mlex\_model\_registry: 65b803a  
-	mlex\_api: f7a4dd6.   
-  mlex\_dash\_segmentation\_demo

Then build the images of the models using the command `make build_docker`. (Currently fully supports random forest, pyMSDtorch, and kmeans model) 

It supports asynchronous job submissions and results showing: choose which (completed) training results from the list to use for segmenting. Similarly, choose which (completed) segmenting (deploy) results to show.
 
