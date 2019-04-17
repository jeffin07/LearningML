from PIL import Image
import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
# import vgg_tf


class Dataset:
	def __init__(self,dataset_path):
		self.dataset_path = dataset_path
		self.cats = []
		self.data_path = []
	def read_dataset(self):
		print("reading dataset")
		dataset_path = self.dataset_path
		label_names = open(os.path.join(dataset_path, "cat.names"), "w")
		
		for cats in os.listdir(os.path.join(dataset_path,"images")):
			# split=0
			good_images = []
			self.cats.append(cats)
			label_names.write(cats+"\n")
			for images in os.listdir(os.path.join(dataset_path, "images", cats)):
				# split+=1
				try:
					print("Processing image {} ".format(images))
					img = Image.open(os.path.join(dataset_path, "images", cats, images))
					if img.mode == "RGB":
						good_images.append(os.path.join(dataset_path, "images", cats, images))

				except Exception as e:
					raise e
					continue
			self.data_path.append(good_images)
	
	def resize_image(self,img, size):
		h,w  = img.shape[:2]
		sh, sw = size
		if h > sh or w > sw:  # shrinking image
			interp = cv2.INTER_AREA
		else: # stretching image
			interp = cv2.INTER_CUBIC
		new_img = cv2.resize(img, size, interpolation=interp)
		return new_img

	def get_batch(self, batch_size=4):
		# print(self.cats)
		images = []
		labels = []
		empty = False
		counter = 0
		each_cat_no = batch_size/len(self.cats)
		while True:
			for i in range(len(self.cats)):
				label = np.zeros(len(self.cats),dtype=int)
				label[i] = 1
				if len(self.data_path[i]) < counter+1:
					empty = True
					continue
				empty=False
				img = cv2.imread(self.data_path[i][counter])
				img = self.resize_image(img,(224,224))
				images.append(img)
				labels.append(label)
			counter+=1

			if empty:
				break
			if (counter) % each_cat_no == 0:
				yield np.array(images,dtype=np.uint8),np.array(labels,dtype=np.uint8)
				del images
				del labels
				images = []
				labels = []






