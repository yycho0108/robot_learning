#!/usr/bin/env python2
import rospy
import rospkg

import tf
import numpy as np
import os
import cv2

from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA, Header
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
from functools import partial
from utils import anorm

try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

class DataCollector(object):
    """Handles ROS I/O communications and conversions.

    Note:
        Although this class handles most ROS-related code,
        all nodes using this class will need to manually call rospy.init_node()
        to prevent possible conflicts in complex-node scenarios.

    Attributes:
        scan_ (np.ndarray): [N,2] Filtered array of (h,r)
        pose_ (np.ndarray, optional): [3]  Latest pose, (x,y,h)
            note that pose_ is only supported when use_tf:=False.
    """
    def __init__(self, start=True,
            use_tf=True, sync=True, slop=0.05,
            min_dr=0.01, min_dh=np.deg2rad(1.0),
            collect_scan=False
            ):
        """
        Args:
            use_tf(bool): Get Odom from tf (default : True)
            sync(bool): Synchronize Odom-Scan Topics (only enabled if use_tf=False, default: True)
            slop(float): Sync slop window (only enabled if sync:=True, default : 0.05)
        """
        self.use_tf_ = use_tf
        self.sync_   = (not use_tf) and sync
        self.slop_   = slop
        self.min_dr_ = min_dr
        self.min_dh_ = min_dh
        self.collect_scan_ = collect_scan

        # Data
        self.t0_   = None
        self.time_ = None
        self.img_  = None
        self.odom_ = None
        self.scan_ = None
        self.dataset_ = []
        self.new_data_ = False
        self.last_data_ = None

        # ROS Handles
        self.scan_sub_ = None
        self.odom_sub_ = None
        self.img_sub_  = None
        self.sync_sub_ = None # for synchronized version
        self.tfl_ = None
        self.br_  = CvBridge()

        if start:
            self.start()

    def img_cb(self, msg):
        """ store img msg """
        self.img_ = msg

    def scan_cb(self, msg):
        """ store scan msg """
        self.scan_= msg

    def odom_cb(self, msg):
        """ store odom msg """
        self.odom_ = msg

    def data_cb(self, *args):
        """ store synced scan/odom msg """
        if self.collect_scan_:
            img_msg, odom_msg, scan_msg = args
        else:
            img_msg, odom_msg = args

        time = odom_msg.header.stamp.to_sec()
        if self.t0_ is None:
            self.t0_ = time
        # track relative time
        self.time_ = (time - self.t0_)
        self.img_cb(img_msg)
        if self.collect_scan_:
            self.scan_cb(scan_msg)
        self.odom_cb(odom_msg)
        self.new_data_ = True

    def start(self):
        """ register ROS handles and subscribe to all incoming topics """
        if self.sync_:
            if self.collect_scan_:
                scan_sub = message_filters.Subscriber('/stable_scan', LaserScan) # TODO: might work well?
            odom_sub = message_filters.Subscriber('/odom', Odometry) 
            img_sub  = message_filters.Subscriber('/camera/image_raw', Image) 
            subs = [img_sub, odom_sub]
            if self.collect_scan_:
                subs.append(scan_sub)
            self.sync_sub_ = message_filters.ApproximateTimeSynchronizer(
                    subs, 10, self.slop_, allow_headerless=False)
            self.sync_sub_.registerCallback(self.data_cb)
        else:
            self.scan_sub_ = rospy.Subscriber('scan', LaserScan, self.scan_cb)
            if not self.use_tf_:
                self.odom_sub_ = rospy.Subscriber('odom', Odometry, self.odom_cb)
        self.tfl_ = tf.TransformListener()

    def convert_translation_rotation_to_pose(self, p, q):
        return Pose(position=Point(*p),
                orientation=Quaternion(x=q[0],
                    y=q[1],
                    z=q[2],
                    w=q[3]))


    def convert_pose_to_xy_and_theta(self, pose):
        """ Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple """
        orientation_tuple = (pose.orientation.x,
                             pose.orientation.y,
                             pose.orientation.z,
                             pose.orientation.w)
        angles = tf.transformations.euler_from_quaternion(orientation_tuple)
        return (pose.position.x, pose.position.y, angles[2])

    @property
    def odom(self):
        """ Get latest pose from TF/Cached Msg.

        Returns:
            odom(np.ndarray): None if odom is not available yet;
                otherwise [3] array formatted [x,y,h]
        """
        pose_msg = None
        if self.use_tf_:
            try:
                #pose_tf = self.tfl_.lookupTransform('base_link', 'odom', rospy.Time(0))
                pose_tf = self.tfl_.lookupTransform('odom', 'base_link', rospy.Time(0))
            except tf.Exception as e:
                rospy.loginfo_throttle(1.0, 'Failed TF Transform : {}'.format(e) )
                return None
            pose_msg = self.convert_translation_rotation_to_pose(*pose_tf)
        else:
            if self.odom_ is not None:
                pose_msg = self.odom_.pose.pose
                self.odom_ = None # clear msg
        try:
            if pose_msg is not None:
                x,y,h = self.convert_pose_to_xy_and_theta(pose_msg)
                return np.asarray([x,y,h])
        except Exception as e:
            rospy.loginfo_throttle(1.0, 'Getting Odom information failed : {}'.format(e))
            return None

    @property
    def img(self):
        msg = self.img_
        if msg is not None:
            try:
                img = self.br_.imgmsg_to_cv2(self.img_, desired_encoding='bgr8')
                #img = cv2.resize(img, (320,240)) # produce half-size image
                return img
            except CvBridgeError as e:
                rospy.loginfo_throttle(1.0, 'Image Conversion Failed : {}'.format(e))
                return None
        return None

    @property
    def scan(self):
        """ Get latest scan data.

        Note:
            assumes scan angle corresponds to (0-2*pi)

        Returns:
            scan(np.ndarray): None is scan is not available yet;
                otherwise [N,2] array formatted [angle,range]
        """
        msg = self.scan_
        if msg is not None:
            self.scan_ = None # clear msg
            #angles = anorm(np.linspace(0, 2*np.pi, len(msg.ranges), endpoint=True))
            #angles = msg.angle_min + np.arange(msg.ranges)*msg.angle_increment
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            mask   = (msg.range_min < ranges) & (ranges < msg.range_max) & np.isfinite(ranges)
            ranges[~mask] = 0.0
            return np.stack([ranges,mask], axis=-1)
            #return np.stack([angles[mask], ranges[mask]], axis=-1)
        else:
            #rospy.loginfo_throttle(1.0, "No scan msg available")
            return None

    @property
    def data(self):
        """ Get latest data+scan

        Returns:
            odom,scan : refer to DataCollector.odom() and DataCollector.scan().
        """
        if self.collect_scan_:
            return (self.time_, self.img, self.odom, self.scan)
        else:
            return (self.time_, self.img, self.odom)

    def append(self, data):
        check = [(e is not None) for e in data]
        rospy.loginfo_throttle(1.0, 'check : {}'.format(check))
        if np.alltrue(check):
            if (self.last_data_ is not None):
                # check against last data for minimum motion
                (x0,y0,h0) = self.last_data_[2]
                (x1,y1,h1) = data[2]
                dr = np.linalg.norm([x1-x0, y1-y0])
                dh = anorm(np.abs(h1-h0))
                if(dr <= self.min_dr_ and dh <= self.min_dh_):
                    return
            self.dataset_.append(data)
            self.last_data_ = data

    def reset(self):
        self.time_ = None
        self.img_  = None
        self.odom_ = None
        self.scan_ = None

    def save(self, path='/tmp'):
        Path(path).mkdir(parents=True, exist_ok=True)
        if len(self.dataset_) <= 0:
            print('No Data To Save')
            return

        data = zip(*self.dataset_) # reformat to [time, img, odom, scan]
        dtypes = [np.float32, np.uint8, np.float32]
        if self.collect_scan_:
            dtypes.append( np.float32 )
        data = [np.asarray(d, dtype=t) for (d,t) in zip(data, dtypes)]
        stamp, img, odom = data[:3]
        if self.collect_scan_:
            scan = data[-1]

        # cannot call .savez due to bug (https://stackoverflow.com/questions/25552741/python-numpy-not-saving-array)
        np.save(os.path.join(path,'stamp.npy'), stamp)
        np.save(os.path.join(path,'img.npy'), img)
        np.save(os.path.join(path,'odom.npy'), odom)
        if self.collect_scan_:
            np.save(os.path.join(path,'scan.npy'), scan)

def main():
    rospy.init_node('data_collector')

    # roll params
    use_tf = rospy.get_param('~use_tf', False)
    sync   = rospy.get_param('~sync', True)
    slop   = rospy.get_param('~slop', 0.01)
    rate   = rospy.get_param('~rate', 10.0)

    min_dr = rospy.get_param('~min_dr', 0.01) #min of 1cm/1deg for "next data"
    min_dh = rospy.get_param('~min_dh', np.deg2rad(1.0))

    rospack   = rospkg.RosPack() 
    pkg_root  = rospack.get_path('robot_learning') # Gets the package
    data_root = os.path.join(pkg_root, 'data', 'train')

    try:
        run_id = rospy.get_param('~run_id', len(os.listdir(data_root)))
    except Exception as e:
        # TODO : lazy
        run_id = 0

    path   = os.path.join(data_root, str(run_id))

    rospy.loginfo('== Parameters ==')
    rospy.loginfo('use_tf : {}'.format(use_tf))
    rospy.loginfo('sync   : {}'.format(sync))
    rospy.loginfo('slop   : {}'.format(slop))
    rospy.loginfo('rate   : {}'.format(rate))
    rospy.loginfo('path   : {}'.format(path))
    rospy.loginfo('================')

    dc = DataCollector(start=True,
            use_tf=use_tf, sync=sync, slop=slop,
            min_dr=min_dr,
            min_dh=min_dh
            )

    save_fn = partial(dc.save, path=path)
    rospy.on_shutdown(save_fn) # save on shutdown
    rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        if dc.new_data_:
            dc.new_data_ = False
            dc.append(dc.data)
            dc.reset()
        rospy.loginfo_throttle(1.0, 'Current Dataset Length : {}'.format(len(dc.dataset_)))
        rate.sleep()

if __name__ == "__main__":
    main()
