#!/usr/bin/env python

import time
import itertools
import numpy as np
import cv2


class Camera(object):

	def __init__(self, grid_dims, save_img_path='./', hough_config={}):
		self.grid_dims = grid_dims
		self.save_img_path = save_img_path
		self.hough_config = hough_config

	def order_circles(self, circles):
		# order the circle objects from left-to-right
		# top-to-bottom
		circles = circles[0]
		rows = np.split( circles[circles[:,1].argsort()], self.grid_dims[0])
		sorted_rows = []
		for row in rows:
			sorted_rows.append(row[row[:,0].argsort()])
		return np.vstack(sorted_rows).reshape((1, circles.shape[0], 3))

	@staticmethod
	def rgb_to_bgr(rgb):
		return (rgb[2], rgb[1], rgb[0])

	@staticmethod
	def bgr_to_rgb(bgr):
		return (bgr[2], bgr[1], bgr[0])

	@staticmethod
	def rgb_to_hex(rgb):
		return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

	@staticmethod
	def bgr_to_hex(bgr):
		rgb = bgr_to_rgb(bgr)
		return rgb_to_hex(rgb)

	@staticmethod
	def hex_to_rgb(hex_):
		if '#' in hex_:
			hex_ = hex_.lstrip('#')
		return tuple(int(hex_[i:i+2], 16) for i in (0, 2, 4))


	def detect_circles(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blurred = cv2.medianBlur(gray, 5) 
		circles = cv2.HoughCircles(
						blurred, 
						cv2.HOUGH_GRADIENT,
						1,
						20,
						param1=70, 
						param2=35, 
						minRadius=0, 
						maxRadius=30,
					)
		circles =  np.uint16(np.around(circles))
		print('num circles detected : ', circles.shape[1])
		if self.grid_dims[0]*self.grid_dims[1]==circles.shape[1]:
			satisfied=True
		else:
			satisfied=False
		return circles, satisfied


	def annotate_image(self, circles, iteration, img):
		for ix, circle in enumerate(circles[0,:]):
			if ix == iteration-1:
				color=(0,255,0) 
			else:
				color=(255,255,255) # white
			cv2.circle(img, (circle[0], circle[1]), circle[2], color, 2) # outer circle
			cv2.circle(img, (circle[0], circle[1]), 2, color, 3) # outer circle

		return img

	def assign_pixels(self, circle, img):
		# circle is coors of the circle
		# img is the original image
		mask = np.zeros_like(img)
		mask = cv2.circle(mask, (circle[0],circle[1]), circle[2], (255,255,255), -1)
		mask_flat = mask.reshape((mask.shape[0]*mask.shape[1], 3))
		where = np.where(mask_flat==np.array([255,255,255]))[0]
		assert where.shape[0] < mask_flat.shape[0]
		# project onto original image
		pixels = img.reshape((img.shape[0]*img.shape[1],3))[where]

		return pixels

	def get_avg_color(self, pixels):
		# pixels is an Nx3 array
		bgr = np.uint16(np.around(np.average(pixels, axis=0)))
		return self.bgr_to_rgb(bgr)

	def loss_func(self, target_rgb, rgb):
		# the euclidean distance between the target and 
		# measured rgb value
		return np.linalg.norm( rgb-np.array(target_rgb))


	def make_measurement(self, iteration, target_rgb, save_img=False):
		camera = cv2.VideoCapture(0)

		satisfied = False
		while not satisfied:
			# take 2 images, the first is usually bad (throw away)
			_, __ = camera.read()
			time.sleep(1.5)
			_, img = camera.read()

			if save_img:
				cv2.imwrite(f'{self.save_img_path}/original_img.png', img)

			# TODO: crop image here
			trim_x = [400,250]
			trim_y = [800, 650]
			cropped_img = img[
				trim_x[0]:img.shape[0]-trim_x[1], 
				trim_y[0]:img.shape[1]-trim_y[1]
			]

			if save_img:
				cv2.imwrite(f'{self.save_img_path}/cropped_img.png', cropped_img)

			circles, satisfied = self.detect_circles(cropped_img) # detect circles with Hough transform

		ordered_circles = self.order_circles(circles) 
		# get the circle associated with this iteration
		circle = ordered_circles[0,iteration-1,:]

		pixels = self.assign_pixels(circle, cropped_img)
		avg_meas_rgb = self.get_avg_color(pixels)

		loss = self.loss_func(target_rgb, avg_meas_rgb)
		del(camera)

		if save_img:
			cv2.imwrite(
				f'{self.save_img_path}/annot_img.png', 
				self.annotate_image(ordered_circles, iteration, cropped_img)
			)

		return loss, avg_meas_rgb




if __name__ == '__main__':

	# camera = Camera(grid_dims=(2,3), save_img_path='run_imgs')

	# target_hex = '#7d43ff'
	# target_rgb = Camera.hex_to_rgb(target_hex)

	# print(target_rgb)

	# loss, meas_rgb = camera.make_measurement(iteration=1, target_rgb=target_rgb, save_img=True)

	# print('loss :', loss)
	# print('target hex : ', target_hex)
	# print('target rgb : ', target_rgb)
	# print('meas rgb : ', meas_rgb  )

	# take a picture

	camera = cv2.VideoCapture(0)

	_, __ = camera.read()
	time.sleep(1.5)
	_, img = camera.read()

	cv2.imwrite('test_pic.png', img)

	



