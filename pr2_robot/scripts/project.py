#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals

from features import compute_color_histograms
from features import compute_normal_histograms

from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2:
    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    vox = pcl_data.make_voxel_grid_filter()

    LEAF_SIZE = 0.005

    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()

    # PassThrough Filter in z axis
    passthrough = cloud_filtered.make_passthrough_filter()

    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.61
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough.filter()

    # PassThrough Filter in y axis
    passthrough_2 = cloud_filtered.make_passthrough_filter()

    filter_axis_2 = 'y'
    passthrough_2.set_filter_field_name(filter_axis_2)
    axis_min_2 = -0.425
    axis_max_2 = 0.425
    passthrough_2.set_filter_limits(axis_min_2, axis_max_2)

    cloud_filtered = passthrough_2.filter()


    # RANSAC Plane Segmentation
    max_distance = 0.005

    seg = cloud_filtered.make_segmenter()

    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment()


    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)


    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.010)
    ec.set_MinClusterSize(250)
    ec.set_MaxClusterSize(3550)
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

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


# Exercise-3: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        # pcl_cluster = cloud_out.extract(pts_list)
        pcl_cluster = extracted_outliers.extract(pts_list)
        
        # convert the cluster from pcl to ROS using helper function
        sample_cloud = pcl_to_ros(pcl_cluster)
        
        # Extract histogram features
        chists = compute_color_histograms(sample_cloud, using_hsv=True)
        normals = get_normals(sample_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
     
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = sample_cloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Invoking the pr2_mover() 
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    dict_list = []
    place_positions = []
    place_groups = []
    place_names = []
    object_names = []
    object_groups = []

    # Assign the scene number
    test_scene_num = Int32()
    test_scene_num.data = int(1)

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    place_pose_list =  rospy.get_param('/dropbox')

    # Parse parameters into individual variables
    for i in range(0,len(place_pose_list)):
        place_position = place_pose_list[i]['position']
        place_group = place_pose_list[i]['group']
        place_name = place_pose_list[i]['name']
        place_positions.append(place_position)
        place_groups.append(place_group)
        place_names.append(place_name)


    for i in range(0, len(object_list_param)):
        object_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        object_names.append(object_name)
        object_groups.append(object_group)

    # Loop through the detected object list
    for obj in object_list:
        # Get the name (label) of the object
        lbl = obj.label

        # Find the group of the dropbox
        obj_ix = object_names.index(lbl)
        object_group = object_groups[obj_ix]

        # Find the position and name of the dropbox
        place_ix = place_groups.index(object_group)
        place_position = place_positions[place_ix]
        place_name = place_names[place_ix]

        # Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(obj.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        
        # Assign the name of the detected object
        object_name = String()
        object_name.data = str(lbl)
    
        # Create 'pick_pose' for the object based on the centroid
        pick_pose = Pose()
        pick_pose.position.x = float(centroid[0])
        pick_pose.position.y = float(centroid[1])
        pick_pose.position.z = float(centroid[2])
        
        # Assign the arm to be used for pick_place
        arm_name = String()
        arm_name.data = str(place_name)

        # Create 'place_pose' for the object
        place_pose = Pose()
        place_pose.position.x = float(place_position[0])
        place_pose.position.y = float(place_position[1]) 
        place_pose.position.z = float(place_position[2])

        rospy.loginfo('Name: {} - Group: {} - Arm name: {}'.format(object_name.data, object_group, arm_name.data))
        
        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        
        dict_list.append(yaml_dict)


        # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')

        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        #     # TODO: Insert your message variables to be sent as a service request
        #     # resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
        #     resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

        #     print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"
    send_to_yaml(yaml_filename, dict_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('project', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
