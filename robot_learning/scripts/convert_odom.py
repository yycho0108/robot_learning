import rospy
import tf
import numpy as np

from nav_msgs.msg import Odometry
from tf_conversions import posemath as pm

class ConvertOdom(object):
    def __init__(self):
        self.method_ = rospy.get_param('~method', default='tf')
        self.pub_tf_ = rospy.get_param('~pub_tf', default=False)

        # data
        self.odom_ = None
        self.recv_ = False
        self.T_ = None
        self.cvt_msg_ = Odometry()

        # create ROS handles
        self.tfl_ = tf.TransformListener()
        self.tfb_ = tf.TransformBroadcaster()
        self.sub_ = rospy.Subscriber('/android/odom', Odometry, self.odom_cb)
        self.pub_ = rospy.Publisher('cvt_odom', Odometry, queue_size=10)

        self.init_tf()

    def init_tf(self):
        # obtain base_link -> android transform
        try:
            rospy.loginfo_throttle(1.0, 'Attempting to obtain static transform ...')
            #self.tfl_.waitForTransform('base_link', 'android', rospy.Duration(0.5))
            txn, qxn = self.tfl_.lookupTransform('base_link', 'android', rospy.Time(0))
            self.T_ = self.tfl_.fromTranslationRotation(txn,qxn)
            self.Ti_ = tf.transformations.inverse_matrix(self.T_)
        except tf.Exception as e:
            rospy.logerr_throttle(1.0, 'Obtaining Fixed Transform Failed : {}'.format(e))

    def odom_cb(self, msg):
        self.odom_ = msg
        self.recv_ = True

    def step(self):
        if self.T_ is None:
            self.init_tf()
            return

        if self.recv_:
            self.recv_ = False
            pose = self.odom_.pose.pose
            T0 = pm.toMatrix(pm.fromMsg(pose)) # android_odom --> android

            # send odom -> base_link transform
            T = tf.transformations.concatenate_matrices(self.T_,T0,self.Ti_)
            frame = pm.fromMatrix(T)
            if self.pub_tf_:
                txn, qxn = pm.toTf(frame)
                self.tfb_.sendTransform(txn, qxn, self.odom_.header.stamp,
                        'odom',
                        'base_link'
                        )

            # send as msg
            # TODO : does not deal with twist/covariance information
            msg = pm.toMsg(frame)
            self.cvt_msg_.pose.pose = pm.toMsg(frame)
            self.cvt_msg_.header.frame_id = 'map' # experimental
            self.cvt_msg_.header.stamp = self.odom_.header.stamp
            self.pub_.publish(self.cvt_msg_)

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()

def main():
    rospy.init_node('convert_odom')
    node = ConvertOdom()
    node.run()

if __name__ == "__main__":
    main()
