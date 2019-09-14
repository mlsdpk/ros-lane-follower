#! /usr/bin/env python

'''
  Ros Node for publishing twist messages based on lane lines detection
'''

# importing necessary packages and modules
from __future__ import division
import rospy
import time
import cv2
import numpy as np
#import pickle as plk
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from dynamic_reconfigure.server import Server
from lane_detect_follower.cfg import CteControllerConfig
from lane_detection import LaneDetection

class LaneDetectFollower(object):

    def __init__(self):

        # Publisher for twist messages
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        # Subscribe to camera messages
        self.image_sub     = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_callback)

        # Create Dynamic Reconfigure Server for PID gains
        self.server = Server(CteControllerConfig, self.dynamic_reconfig_callback)

        self.Kp = 0.01
        self.Ki = 0
        self.Kd = 0.05

        self.wheel_base = 0.188 # m
        self.wheel_radius = 0.0325

        self.max_desire_linear_vel = 0.3 # m/s
        self.max_omega = 33.51 # rad/s
        self.last_cte = 0
        self.angle_const = 0.01

        self.debug = False
        self.counter = 1

        # set to True if you want to manually update the PID gains
        self.manual_update_pid = False

        # following block of code is for camera calibration

        ###################################################
        # Need to update the pickle file (change default protocol to 2)
        # pickle.dump(your_object, your_file, protocol=2)
        # Because Python 2 can only read value of 2
        ###################################################

        # dist_pick = plk.load(open(rospy.get_param('~path'),'rb'))
        # self.mtx = dist_pick['mtx']
        # self.dist = dist_pick['dist']

    def dynamic_reconfig_callback(self, config, level):

        if self.manual_update_pid:
            self.Kp = config["Kp"]
            self.Ki = config["Ki"]
            self.Kd = config["Kd"]

        return config

    # Main Callback Function
    def camera_callback(self, data):

        # Dividing frame rate by 3 (10fps)
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        try:
            #### direct conversion to CV2 ####
            np_arr = np.fromstring(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        except:
            print("Error conversion to CV2")

        # Undistorted image
        # undist = cv2.undistort(cv_image, self.mtx, self.dist, None, self.mtx)

        # Create lane detection object
        lane_detection_object = LaneDetection()

        # Lane lines detection and process cross track error
        cte, angle, final_img = lane_detection_object.processImage(cv_image)

        cmd_vel = Twist()

        if cte is not None and angle is not None:
            angular_z = self.Kp * cte + self.Kd * (cte - self.last_cte)
            self.last_cte = cte
            linear_x = self.max_desire_linear_vel
            angular_z = max(angular_z, -2.0) if angular_z < 0 else min(angular_z, 2.0)
        else:
            angular_z = self.Kp * self.last_cte * 1.9
            linear_x = self.max_desire_linear_vel
            angular_z = max(angular_z, -2.0) if angular_z < 0 else min(angular_z, 2.0)

        cmd_vel.linear.x = linear_x
        cmd_vel.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd_vel)

        if self.debug:
            w_l = ((2 * linear_x)-(angular_z * self.wheel_base)) / (2 * self.wheel_radius)
            w_r = ((2 * linear_x)+(angular_z * self.wheel_base)) / (2 * self.wheel_radius)

            pwm_l = int((255 * w_l)/self.max_omega)
            if abs(pwm_l) > 255:
                pwm_l = 255
            elif abs(pwm_l) < 0:
                pwm_l = 0
            else:
                pwm_l = abs(pwm_l)


            pwm_r = int((255 * w_r)/self.max_omega)
            if abs(pwm_r) > 255:
                pwm_r = 255
            elif abs(pwm_r) < 0:
                pwm_r = 0
            else:
                pwm_r = abs(pwm_r)

            print("w_l : " + str(w_l))
            print("pwm_l : " + str(pwm_l))
            print("w_r : " + str(w_r))
            print("pwm_r : " + str(pwm_r))
            print("----------")

        if lane_detection_object.draw_img:
            cv2.imshow("Original Image", final_img)
            cv2.waitKey(1)

    def clean_up(self):

        cv2.destroyAllWindows()

def main():

    rospy.init_node("lane_detect_follower_node", anonymous=True)

    lane_detect_follower_object = LaneDetectFollower()

    rate = rospy.Rate(5)
    ctrl_c = False

    def shutdownhook():

        lane_detect_follower_object.clean_up()
        ctrl_c = True

    rospy.on_shutdown(shutdownhook)

    while not ctrl_c:
        rate.sleep()



if __name__ == '__main__':
    main()
