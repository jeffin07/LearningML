from PIL import Image
import os
import numpy as np 
import cv2
import random

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
		# label_names.close()
		# train_names = open(os.path.join(dataset_path, "train.txt"), "w")
		# test_names = open(os.path.join(dataset_path, "test.txt"), "w")
		# random.shuffle(good_images)
		# split = 0
		# print("splitting train/test")
		# for i in good_images:
		# 	split+=1
		# 	if split == 5:
		# 		test_names.write(i+"\n")
		# 		split=0
		# 	else:
		# 		train_names.write(i+"\n")
		# train_names.close()
		# test_names.close()
		# train_names = open(os.path.join(dataset_path, "train.txt"), "r").readlines()
		# self.train_names = [i.split("\n")[0] for i in train_names]
		# test_names = open(os.path.join(dataset_path, "test.txt"), "r").readlines()
		# self.test_names = [i.split("\n")[0] for i in test_names]

	def get_batch(self, batch_size=4):
		print(self.cats)
		images = []
		labels = []
		empty = False
		counter = 0
		each_cat_no = batch_size/len(self.dataset_path)
		while True:
			for i in range(len(self.cats)):
				label = np.zeros(len(self.cats),dtype=int)
				label[i] = 1
				if len(self.data_path[i]) < counter+1:
					empty = True
					continue
				img = cv2.imread(self.data_path[i][counter])
				images.append(img)
				labels.append(label)
			counter+1

			if empty:
				break
			if counter%batch_size == 0:
				yield np.array(images,dtype=np.uint8),np.array(labels,dtype=np.uint8)
				del images
				del labels
				images = []
				labels = []






sample_data = Dataset('/home/jeffin/git/Me/LearningML/data/train')
sample_data.read_dataset()
for i in range(5):
	k,j = sample_data.get_batch()
	print(k)