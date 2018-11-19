#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist

class Teleop(object):
    """
    Keyboard-based teleoperation node.
    Repeats the last command at an interval of `~period` seconds,
    then stops the robot entirely after `~timeout` seconds since the last command input.
    """
    def __init__(self):
        # handle ros paramters
        self.period_  = rospy.get_param('~period', 0.0)
        self.v_scale_ = rospy.get_param('~v_scale', 1.0)
        self.w_scale_ = rospy.get_param('~w_scale', 1.0)
        self.timeout_ = rospy.get_param('~timeout', 1.0)
        self.res_ = rospy.get_param('~resolution', 255)
        self.viz_ = rospy.get_param('~viz', True)
        self.decay_ = rospy.get_param('~decay', 0.99)

        # actions
        self.amap_ = {
                'q' : [1.1, 1.1],
                'z' : [0.9, 0.9],
                'w' : [1.1, 1.0],
                'x' : [0.9, 1.0],
                'e' : [1.0, 1.1],
                'c' : [1.0, 0.9]
                }

        # cmd gui input
        self.x_, self.y_ = 0,0
        self.cmd_v_ = 0
        self.cmd_w_ = 0
        self.cmd_active_ = False
        self.last_cmd_ = rospy.Time(0)

        # cmd pub output
        self.last_pub_ = rospy.Time(0)
        self.cmd_pub_  = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    def help(self):
        print '\r[params] repeat : {} | period : {}'.format(self.period_>0, self.period_)
        print '\rq/z : increase/decrease max speeds by 10%'
        print '\rw/x : increase/decrease only linear speed by 10%'
        print '\re/c : increase/decrease only angular speed by 10%'

    def mouse_cb(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cmd_active_ = True
            self.x_, self.y_ = x, y
            self.cmd_v_ = (-y) / float(self.res_) + 0.5
            self.cmd_w_ = np.sign(self.cmd_v_) * (-x/float(self.res_) + 0.5)
            self.last_cmd_ = rospy.Time.now()

        if event == cv2.EVENT_LBUTTONUP:
            self.cmd_active_ = False

        if event == cv2.EVENT_MOUSEMOVE:
            if self.cmd_active_:
                self.x_, self.y_ = x, y
                self.cmd_v_ = (-y) / float(self.res_) + 0.5
                self.cmd_w_ = np.sign(self.cmd_v_) * (-x/float(self.res_) + 0.5)
                self.last_cmd_ = rospy.Time.now()

    def run(self):
        """ main loop """
        repeat_flag = (self.period_ > 0)
        cmd_vel     = Twist()

        self.help()

        viz = np.zeros(shape=(self.res_, self.res_), dtype=np.float32)
        viz[:,self.res_/2] = 1.0
        viz[self.res_/2,:] = 1.0

        acc = np.zeros_like(viz)
        cv2.namedWindow('control')
        cv2.setMouseCallback('control', self.mouse_cb)
        cv2.imshow('control', viz)

        try:
            while not rospy.is_shutdown():
                # current time (technically has some delay)
                now = rospy.Time.now()

                # handle key input
                k = cv2.waitKey(10)
                if k == 27:
                    break
                if k in self.amap_:
                    self.new_key_ = False
                    sv, sw = self.amap_[k]
                    self.v_scale_ *= sv
                    self.w_scale_ *= sw
                    rospy.loginfo('(v,w) = ({},{})'.format(
                        self.v_scale_, self.w_scale_))

                # convert control input to velocity commands
                cmd_vel.linear.x = self.cmd_v_ * 2 * self.v_scale_
                cmd_vel.angular.z = self.cmd_w_ * 2 * self.w_scale_

                if self.viz_:
                    acc[self.y_-1:self.y_+2,self.x_-1:self.x_+2] += 0.1
                    #cv2.circle(acc, (self.x_, self.y_), 3, 1.0, -1)
                    acc *= self.decay_
                    cv2.imshow('control', viz + acc)

                # handle timeout
                if not self.cmd_active_:
                    if (now - self.last_cmd_).to_sec() > (self.timeout_):
                        cmd_vel.linear.x  = 0
                        cmd_vel.angular.z = 0

                # handle publishing
                if repeat_flag and (now - self.last_pub_).to_sec() > (self.period_):
                    self.cmd_pub_.publish(cmd_vel)
                    self.last_pub_ = now
        except Exception as e:
            rospy.loginfo('{}'.format(e))
        finally:
            cmd_vel.linear.x = cmd_vel.angular.z = 0.0
            self.cmd_pub_.publish(cmd_vel)


def main():
    rospy.init_node('cv_mouse_teleop')
    node = Teleop()
    node.run()

if __name__ == "__main__":
    main()

