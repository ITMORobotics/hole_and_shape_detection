#!/usr/bin/env python3

# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import sys
import time
import rospy

import numpy.linalg as LA

from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import *

J_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

import median_filter as md
from realsense_device_manager import DeviceManager
from graphics_manager import GraphicRealsense

from threading import Thread

median_circ = md.MedianFilter(3, 15)
median_sqr = md.MedianFilter(3, 15)

def send_pose(publisher, xyz: np.ndarray):
	pose = Pose()
	pose.position.x = xyz[0]
	pose.position.y = xyz[1]
	pose.position.z = xyz[2]
	publisher.publish(pose)

class myThread (Thread):
	counter=0
	def __init__(self, w, device_manager):
		Thread.__init__(self)
		self.w = w
		self.device_manager = device_manager
		self.counter = 0

	def run(self):
		try:
			self.device_manager.enable_all_devices()
			while True:
				framesets = self.device_manager.wait_align_framesets()
				img_orig = list(device_manager.framesets_to_imgs(framesets).values())[0]
				depth_img = list(device_manager.framesets_to_depths(framesets, color = True).values())[0]
				XYZ = list(device_manager.framesets_to_points(framesets).values())[0]
				img_circ, circle = detecCircle(img_orig)
				img_sqr, square = detecRectangle(img_orig)
				if not np.all(circle == 0):
					img, XYZ_circ = getXYZCircle(circle, img_circ, depth_img, XYZ)
					if (XYZ_circ[2] != 0):
						XYZ_circ = median_circ.apply_median(XYZ_circ)

						if not np.all(XYZ_circ == 0):
							send_pose(circle_publisher, np.array(XYZ_circ).flatten())
				if not np.all(square == 0):
					img , XYZ_sqr = getXYZRectangle(square, img_sqr, depth_img, XYZ)
					if (XYZ_sqr[2] != 0):
						XYZ_sqr = median_sqr.apply_median(XYZ_sqr)

						if not np.all(XYZ_sqr == 0):
							send_pose(square_publisher, np.array(XYZ_sqr).flatten())
				w.updateIMG(img_sqr)
				if rospy.is_shutdown():
					break

		except KeyboardInterrupt:
			print("The program was interupted by the user. Closing the program...")

		finally:
			self.device_manager.disable_streams()

def getXYZCircle(circle, img, depth, XYZ):
	offset = 25
	circle_offset = 5
	R = circle[2]
	depth_original = np.copy(depth)

	XYZ = XYZ.reshape(img.shape[0], img.shape[1], 3)

	XYZ = XYZ[circle[1]-R-offset:circle[1]+R+offset, circle[0]-R-offset:circle[0]+R+offset]
	img = img[circle[1]-R-offset:circle[1]+R+offset, circle[0]-R-offset:circle[0]+R+offset]
	depth = depth[circle[1]-R-offset:circle[1]+R+offset, circle[0]-R-offset:circle[0]+R+offset]
	depth_original = depth_original[circle[1]-R-offset:circle[1]+R+offset, circle[0]-R-offset:circle[0]+R+offset]

	cv.circle(depth, (int(depth.shape[0]/2), int(depth.shape[1]/2)), circle[2]+circle_offset, (0, 0, 0), -1)
	cv.circle(depth, (int(depth.shape[0]/2), int(depth.shape[1]/2)), circle[2]+circle_offset, (0, 0, 0), -1)
	cv.circle(depth_original, (int(depth.shape[0] / 2), int(depth.shape[1] / 2)), circle[2], (128, 0, 0), 2)

	norm = np.sum(depth, axis=2)

	indices = np.array(np.where(norm != 0))
	plane_XYZ = XYZ[indices[0], indices[1]]

	X = np.c_[plane_XYZ, np.ones(plane_XYZ.shape[0])]
	Y = plane_XYZ[:, 2]
	try:
		theta = LA.inv(X.T.dot(X)).dot(X.T).dot(Y)
	except Exception as e:
		print(e)
		return img,[0,0,0]
	A, B, C, D = theta
	C -= 1

	X_circle = (np.max(plane_XYZ[:, 0]) + np.min(plane_XYZ[:, 0])) / 2
	Y_circle = (np.max(plane_XYZ[:, 1]) + np.min(plane_XYZ[:, 1])) / 2
	Z_circle = (-D-A*X_circle - B*Y_circle)/C

	return  depth_original, [X_circle, Y_circle, Z_circle] #X_circle_depth, Y_circle_depth, Z_circle_depth

def getXYZRectangle(square, img, depth, XYZ):
	offset = 25
	square_offset = 5
	depth_original = np.copy(depth)

	XYZ = XYZ.reshape(img.shape[0], img.shape[1], 3)

	XYZ = XYZ[square[1]-offset:square[1]+square[3]+offset, square[0]-offset:square[0]+square[2]+offset]
	img = img[square[1]-offset:square[1]+square[3]+offset, square[0]-offset:square[0]+square[2]+offset]
	depth = depth[square[1]-offset:square[1]+square[3]+offset, square[0]-offset:square[0]+square[2]+offset]
	depth_original = depth_original[square[1]-offset:square[1]+square[3]+offset, square[0]-offset:square[0]+square[2]+offset]

	cv.rectangle(depth, (square[0] - square_offset, square[1] - square_offset), (square[0] + square[2] + square_offset, square[1] + square[3] + square_offset), (0, 0, 0), -1)

	norm = np.sum(depth, axis=2)

	indices = np.array(np.where(norm != 0))
	plane_XYZ = XYZ[indices[0], indices[1]]

	X = np.c_[plane_XYZ, np.ones(plane_XYZ.shape[0])]
	Y = plane_XYZ[:, 2]
	try:
		theta = LA.inv(X.T.dot(X)).dot(X.T).dot(Y)
	except Exception as e:
		print(e)
		return img,[0,0,0]
	A, B, C, D = theta
	C -= 1

	X_circle = (np.max(plane_XYZ[:, 0]) + np.min(plane_XYZ[:, 0])) / 2
	Y_circle = (np.max(plane_XYZ[:, 1]) + np.min(plane_XYZ[:, 1])) / 2
	Z_circle = (-D-A*X_circle - B*Y_circle)/C

	return  depth, [X_circle, Y_circle, Z_circle] #X_circle_depth, Y_circle_depth, Z_circle_depth

def detecRectangle(img):
	img = cv.medianBlur(img, 3)
	hsvFrame = cv.cvtColor(img, cv.COLOR_RGB2HSV)

	blue_lower = np.array([85, 150, 150], np.uint8)
	blue_upper = np.array([115, 255, 255], np.uint8)
	blue_mask = cv.inRange(hsvFrame, blue_lower, blue_upper)

	res_blue = cv.bitwise_and(img, img, mask=blue_mask)

	# convert image into greyscale mode
	gray_image = cv.cvtColor(res_blue, cv.COLOR_BGR2GRAY)

	# find threshold of the image
	_, thrash = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY)
	thrash = cv.bitwise_not(thrash)
	contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	min_w = 10
	min_h = min_w

	max_w = 100
	max_h = max_w

	coordinates = []
	shapes = []
	try:
		for contour in contours:
			shape = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
			if len(shape) == 4:
				coordinates.append(cv.boundingRect(shape))
				shapes.append(shape)

		coordinates = np.array(coordinates)
		shapes = np.array(shapes)

		condw1 = coordinates[:, 2] < max_w
		condw2 = coordinates[:, 2] > min_w
		condh1 = coordinates[:, 2] < max_h
		condh2 = coordinates[:, 2] > min_h
		condsqr1 = coordinates[:, 2] / coordinates[:, 3] >= 0.9
		condsqr2 = coordinates[:, 2] / coordinates[:, 3] <= 1.1

		indices = np.array(condw1 & condw2 & condh1 & condh2 & condsqr1 & condsqr2)

		square = np.copy(coordinates[indices])
		m = (square[:, 2] * square[:, 3]).argsort()[-1]
		square = square[m]

		points = shapes[indices]

		cv.drawContours(img, [points[m]], 0, (0, 255, 0), 4)
	except Exception as e:
		print("Square not found or", e)
		square = [0,0,0,0]
	finally:
		return np.array(img), square

def detecCircle(img):
	img = cv.medianBlur(img, 3)
	hsvFrame = cv.cvtColor(img, cv.COLOR_RGB2HSV)

	blue_lower = np.array([85, 150, 72], np.uint8) # [85, 150, 72]
	blue_upper = np.array([115, 255, 255], np.uint8)
	blue_mask = cv.inRange(hsvFrame, blue_lower, blue_upper)

	# kernal = np.ones((5, 5), "uint8")
	# blue_mask = cv.dilate(blue_mask, kernal)
	res_blue = cv.bitwise_and(img, img, mask=blue_mask)
	gray = cv.cvtColor(res_blue, cv.COLOR_BGR2GRAY)
	try:
		circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=50)
		circles = np.uint16(np.around(circles))
		circles = circles[circles[:, :, 2].argsort()[0, -1]][0]
		# draw the outer circle
		cv.circle(img, (circles[0], circles[1]), circles[2], (0, 255, 0), 2)
		# draw the center of the circle
		cv.circle(img, (circles[0], circles[1]), 2, (0, 0, 255), 3)

		line_length = 20
		cv.line(img, (circles[0]-line_length, circles[1]), (circles[0]+line_length, circles[1]), (0, 0, 255), 1)
		cv.line(img, (circles[0], circles[1]- line_length), (circles[0] , circles[1]+ line_length), (0, 0, 255), 1)

		cv.line(img, (int(img.shape[1]/2) - line_length, int(img.shape[0]/2)), (int(img.shape[1]/2) + line_length, int(img.shape[0]/2)), (255, 0, 0), 1)
		cv.line(img, (int(img.shape[1]/2) , int(img.shape[0]/2)- line_length), (int(img.shape[1]/2) , int(img.shape[0]/2)+ line_length), (255, 0, 0), 1)
	except Exception as e:
		print("Circle not found or", e)
		circles = [0,0,0]
	finally:
		return np.array(img), circles

if __name__ == '__main__':
	resolution_width = 1280  # pixels
	resolution_height = 720  # pixels
	frame_rate = 15 # fps

	# ROS
	rospy.init_node('node_realsense')
	# Publishers
	js_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
	circle_publisher = rospy.Publisher('/circle_pose', Pose, queue_size=10)
	square_publisher = rospy.Publisher('/square_pose', Pose, queue_size=10)


	try:
		# Enable the streams from all the intel realsense devices
		d435_rs_config = rs.config()
		d435_rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
		# d435_rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
		d435_rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.rgb8, frame_rate)

		# Use the device manager class to enable the devices and get the frames
		device_manager = DeviceManager(rs.context(), d435_rs_config)
		# device_manager.enable_all_devices()
	except KeyboardInterrupt:
		print("realsense start error...")

	app = QApplication(sys.argv)
	w = GraphicRealsense()
	thread1 = myThread(w, device_manager)
	thread1.setDaemon(True)
	thread1.start()
	# w.exec()
	sys.exit(app.exec_())






