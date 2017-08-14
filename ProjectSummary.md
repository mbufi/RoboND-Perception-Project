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
# PassThrough Filter
# PassThrough filter in the z-axis    
passthrough_z = cloud_vox.make_passthrough_filter()

filter_axis_z = 'z'
passthrough_z.set_filter_field_name(filter_axis_z)
passthrough_z.set_filter_limits(0.6, 1.3)

cloud_passthrough = passthrough_z.filter()

# PassThrough Filter in the y axis
passthrough_y = cloud_passthrough.make_passthrough_filter()

filter_axis_y = 'y'
passthrough_y.set_filter_field_name(filter_axis_y)
passthrough_y.set_filter_limits(-0.5, 0.5)

cloud_passthrough = passthrough_y.filter()
```

Lastly, I used a `RANSAC` plane segmenter to identify the table, to allow me to seperate the objects from the table. Therefore I would have one table cloud, and one object cloud. Max_distance of 0.01 worked the best for the RANSAC segmenter. 
```python
# RANSAC Plane Segmentation
max_distance = 0.01

seg = cloud_passthrough.make_segmenter()

seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

seg.set_distance_threshold(max_distance)

inliers, coefficients = seg.segment()


# TODO: Extract inliers and outliers
cloud_table = cloud_passthrough.extract(inliers, negative=False) #table

cloud_objects = cloud_passthrough.extract(inliers, negative=True) #objects on table
```


### 2. Pipeline including clustering for segmentation implemented.  
Once the initial point clouds were worked on to the point of having a particular "object cloud" that represented the objects on the table, the next step was to cluster the points into the different objects that were present. To do this, the color information was removed (white cloud):
```python 
 white_cloud = XYZRGB_to_XYZ(cloud_objects) # Apply function to convert XYZRGB to XYZ
 tree = white_cloud.make_kdtree()
```

and then `Euclidean Clustering` was performed. A lot of experimenting has to go into fine tuning the paramenters ClusterTolerance, MinClusterSize, and MaxClusterSize due to the objects being various sizes. 
```python 
# TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
ec = white_cloud.make_EuclideanClusterExtraction()
# Set tolerances for distance threshold 
# as well as minimum and maximum cluster size (in points)
ec.set_ClusterTolerance(0.015)
ec.set_MinClusterSize(20)
ec.set_MaxClusterSize(3000)

# Search the k-d tree for clusters
ec.set_SearchMethod(tree)
# Extract indices for each of the discovered clusters
cluster_indices = ec.Extract()
#Assign a color corresponding to each segmented object in scene
cluster_color = get_color_list(len(cluster_indices))

color_cluster_point_list = []

for j, indices in enumerate(cluster_indices):
    for i, indice in enumerate(indices):
        color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])

# Create new cloud containing all clusters, each with unique color
cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)
```
After that, the last step is to apply a different color to each detected cluster. The PCL clouds are then converted back to a ROS format and then published to various topics to be seen in RViz. 
```python 
# TODO: Convert PCL data to ROS messages
ros_cloud_objects = pcl_to_ros(cloud_objects)
ros_cloud_table = pcl_to_ros(cloud_table)
ros_cluster_cloud = pcl_to_ros(cluster_cloud)

# TODO: Publish ROS messages
pcl_objects_pub.publish(ros_cloud_objects)
pcl_table_pub.publish(ros_cloud_table)
pcl_cluster_pub.publish(ros_cluster_cloud)
```

### 3. Features extracted and SVM trained.  Object recognition implemented.
Using the lessons, I `implemented the compute_color_histograms` and `compute_normal_histograms` functions. I set the range of [0,256] for the colors, and [-1,1] for the normals. I also used 64 bins. 

Next step was to train my model. When starting, using 20 images of each item provided about ~71% accuracy. I only achieved 90%+ when providing 500+ images. As shown below, the final accuracy of model turned out to be pretty decent. I think if I let it run for a couple thousand images per item, it would be near perfect accuracy. 

The last step was to grab the pretrained SVM model inside the perception pipeline to identify the individual object clusters that were detected during the clustering step. This model labels each cluster, and then I publish these labels into RViz with their name to check if the model is correct in classifying everything. 

See below for the results.




## Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

After all the objects were detected with classifier, `pr2_mover()` function is called to create the output .yaml files. 

The function first reads in the object list parameters .yaml file found in the project, and parses it to find out which objects should be collected in each case. The same thing happens with the dropbox .yaml file to find out where objects should be dropped/placed. 

The main loop then iterates over all the items that need to be collected (determined from the picklist). This part is incharge of setting the object name based on the .yaml data, and converting it to the correct datatype to be understood by ROS. The next step is to determie the `pick_pose`. I initalize all values to zero, before moving onto the next step. It is especially helpful that the position is initalized to zero, since it serves as a way to check that all items that get classified have a proper position assigned to them before creating the output .yaml file. 

The code then checks through the objects that has been detected to see if the correct one does exist, and updates it `pick_pose.positon` with the correct values provided by the centroid of the object. It will then check if the ground specified was red or green, and then sets the correct `arm_name`.

Finally, the `test_scene_number` is specified and the dictionary is created. This is used to make the final output yaml file. 



## Final Results

All final output .yaml files can be seen in the repo. 

World 1 scored 100% (4/4)
World 2 scored 
World 3 scored 

## Conclusion
Overall, my perception pipeline worked pretty well. I know it could be more accurrate with more training samples and tighter tolerances with my parameters...perhaps I will experiment with this later. 

I plan to try the challenges on my spare time and have had great fun with this project.  




