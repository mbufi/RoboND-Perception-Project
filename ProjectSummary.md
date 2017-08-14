# Project: Perception Pick & Place


---

[//]: # "Image References"

[image1a]: ./output/grid.PNG
[image1b]: ./output/warpedGrid.PNG
[image2a]: ./output/colorThresh.PNG
[image3]: ./output/completedThresh.PNG
[image4a]: ./output/MaptoWorld.PNG
[image4b]: ./output/rotationAndTranslation.PNG
[image4c]: ./output/rotationMatrix.PNG
[image5]: ./output/completeAutoRun.PNG
[video1]: ./output/test_mapping.MP4

![alt text][image2a]
### Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  See the example `output.yaml` for details on what the output should look like.  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

### Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!



### 1. Pipeline for filtering and RANSAC plane fitting implemented.

To start this, I grabbed all incoming ROS cloud information and coverted it to PCL (Point cloud) format so that I could use the PCL library on the data. 
```python 
pcl_data = ros_to_pcl(pcl_msg)
```

The `statistical outlier filter` was used to filter out the noise from the camera, and leave a decently clean image. When viewing the image initally without the filter, it was very noisy and I figured it would cause clustering issues. By reducing the std_dev parameter, I managed to get rid of most of the noise and settled with a value of 0.3. This value could go lower, but there reaches a point in which you start losing valuable data needed for classification.
```python
# Statistical Outlier Filtering
# creating a filter object
outlier_filter = pcl_data.make_statistical_outlier_filter()
#Set the number of neighboring points to analyze for any given point
outlier_filter.set_mean_k(20)
#Set threshold scale factor
x = 0.3
#Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
outlier_filter.set_std_dev_mul_thresh(x)
#then call the filter for it to work on the cloud_filtered
outlier_filtered_cloud = outlier_filter.filter()

```

After the `outlier filter`, I implemented a voxel grid filter to downsample the image. This reduces the required amount of computational power required to process and operate on all the point cloud data. After some trial and error, I settled with a LEAF_SIZE of 0.005. This number properly downsampled the image enough, without losing too much detail.
```python
vox = outlier_filtered_cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.005
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud_vox = vox.filter()
```

Next I added a `pass-through filter` in the z-dimension to focus on the area of interest and remove all the other parts of the PCL cloud. I set the min and max parameter to 0.6 amd 1.3 (determined using trial and error). The min parameter has to be increased from the recommended number by the exercises due to not completely removing the table edge. An additional pass-through filter was added in the y-dimension to 
```python

```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)




Here's | A | Snappy | Table
--- | --- | --- | ---
1 | `highlight` | **bold** | 7.41
2 | a | b | c
3 | *italic* | text | 403
4 | 2 | 3 | abcd


### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  












