# Project: Perception Pick & Place


---

[//]: # "Image References"

[image1]: ./images/ModelAccuracy.JPG
[image2]: ./images/world1.JPG
[image3]: ./images/world2.JPG
[image4]: ./images/world3.JPG

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
![alt text][image1]



## Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

After all the objects were detected with classifier, `pr2_mover()` function is called to create the output .yaml files. 

The function first reads in the object list parameters .yaml file found in the project, and parses it to find out which objects should be collected in each case. The same thing happens with the dropbox .yaml file to find out where objects should be dropped/placed. 

The main loop then iterates over all the items that need to be collected (determined from the picklist). This part is incharge of setting the object name based on the .yaml data, and converting it to the correct datatype to be understood by ROS. The next step is to determie the `pick_pose`. I initalize all values to zero, before moving onto the next step. It is especially helpful that the position is initalized to zero, since it serves as a way to check that all items that get classified have a proper position assigned to them before creating the output .yaml file. 

The code then checks through the objects that has been detected to see if the correct one does exist, and updates it `pick_pose.positon` with the correct values provided by the centroid of the object. It will then check if the ground specified was red or green, and then sets the correct `arm_name`.

Finally, the `test_scene_number` is specified and the dictionary is created. This is used to make the final output yaml file. 
```python
# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    dict_list = []
    centroids = [] # to be list of tuples (x, y, z)

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    dict_dropbox = {}
    for p in dropbox_param:
        dict_dropbox[p['name']] = p['position']

    # TODO: Rotate PR2 in place to capture side tables for the collision map
    #Work in progress..
    
    # TODO: Loop through the pick list
    for obj in object_list_param:
        

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        object_name = String()
        object_name.data = obj['name']

        #set default value of pick_pose in case the object can't be found
        pick_pose = Pose()
        pick_pose.position.x = 0
        pick_pose.position.y = 0
        pick_pose.position.z = 0

        #set orientation to 0
        pick_pose.orientation.x = 0
        pick_pose.orientation.y = 0
        pick_pose.orientation.z = 0
        pick_pose.orientation.w = 0

        #set place pose orientation to 0
        place_pose = Pose()
        place_pose.orientation.x = 0
        place_pose.orientation.y = 0
        place_pose.orientation.z = 0
        place_pose.orientation.w = 0

        #print(object_name)
        for detected_object in object_list:
            if detected_object.label == object_name.data:

                # TODO: Create 'place_pose' for the object
                points_arr = ros_to_pcl(detected_object.cloud).to_array()
                pick_pose_np = np.mean(points_arr, axis=0)[:3]
                pick_pose.position.x = np.asscalar(pick_pose_np[0])
                pick_pose.position.y = np.asscalar(pick_pose_np[1])
                pick_pose.position.z = np.asscalar(pick_pose_np[2])
                
                break


        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        if obj['group'] == 'red':
            arm_name.data = 'left'
        elif obj['group'] == 'green':
            arm_name.data = 'right'
        else:
            print "ERROR"


        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        test_scene_num = Int32()
        test_scene_num.data = 3 ## CHANGE THIS for every scene to label a new output yaml

        place_pose.position.x = dict_dropbox[arm_name.data][0]
        place_pose.position.y = dict_dropbox[arm_name.data][1]
        place_pose.position.z = dict_dropbox[arm_name.data][2]
        dict_list.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

    # TODO: Output your request parameters into output yaml file
    yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"

    send_to_yaml(yaml_filename, dict_list)
```


## Final Results

All final output .yaml files can be seen in the repo. 

#### World 1 scored 100% (3/3)
![alt text][image2]


#### World 2 scored 100% (5/5)
![alt text][image3]


#### World 3 scored 100% (8/8)
![alt text][image4]

## Conclusion
Overall, my perception pipeline worked pretty well. I know it could be more accurrate with more training samples and tighter tolerances with my parameters...perhaps I will experiment with this later. 

I plan to try the challenges on my spare time and have had great fun with this project.  




