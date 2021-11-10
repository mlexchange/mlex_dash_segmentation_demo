# Segmentation demo v2.0.0

The fisrt version of image segmentation demo integrated with the model registry and the compute sevice manager.  
To run this demo, `docker-compose` the followings:  
-	model-registry  
-	mlex\_api  
-  dash\_segmentation\_demo

Then build the image of random forest model using command `make build_docker`. (Currently only fully supports random forest model) 

It supports asynchronous job submition and results showing. Choose which (completed) training to use for segment from the list. Simialrly, choose which (completed) segment (deploy) to show results.
 
