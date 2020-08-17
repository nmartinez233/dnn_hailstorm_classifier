# Imports
import random, cv2, os, sys, shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob


class image_clustering:

	def __init__(self, folder_path="data", n_clusters=10, max_examples=None, use_imagenets=False, use_pca=False):
		paths = os.listdir(folder_path)
		if max_examples == None:
			self.max_examples = len(paths)
		else:
			if max_examples > len(paths):
				self.max_examples = len(paths)
			else:
				self.max_examples = max_examples
		self.n_clusters = n_clusters
		self.folder_path = folder_path
		random.shuffle(paths)
		self.image_paths = paths[:self.max_examples]
		self.use_imagenets = use_imagenets
		self.use_pca = use_pca
		del paths 
		try:
			shutil.rmtree("output")
		except FileExistsError:
			pass
		print("\n output folders created.")
		os.makedirs("output")
		for i in range(self.n_clusters):
			os.makedirs("output/cluster" + str(i))
		os.makedirs("output/stitches")
		print("\n Object of class \"image_clustering\" has been initialized.")

	def load_images(self):
		self.images = []
		for image in self.image_paths:
			self.images.append(cv2.cvtColor(cv2.resize(cv2.imread(self.folder_path + image), (256,256)), cv2.COLOR_BGR2RGB))

		self.images = np.float32(self.images)
		self.images /= 255
		print("\n " + str(self.max_examples) + " images from the \"" + self.folder_path + "\" folder have been loaded in a random order.")

	def get_new_imagevectors(self):
		if self.use_imagenets == False:
			self.images_new = self.images
		else:
			if use_imagenets.lower() == "vgg16":
				model1 = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(256,256,3))
			elif use_imagenets.lower() == "vgg19":
				model1 = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(256,256,3))
			elif use_imagenets.lower() == "resnet50":
				model1 = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(256,256,3))
			elif use_imagenets.lower() == "xception":
				model1 = keras.applications.xception.Xception(include_top=False, weights='imagenet',input_shape=(256,256,3))
			elif use_imagenets.lower() == "inceptionv3":
				keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(256,256,3))
			elif use_imagenets.lower() == "inceptionresnetv2":
				model1 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(256,256,3))
			elif use_imagenets.lower() == "densenet":
				model1 = keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(256,256,3))
			elif use_imagenets.lower() == "mobilenetv2":
				model1 = tf.keras.applications.mobilenetv2.MobileNetV2(input_shape=(256,256,3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', pooling=None)
			else:
				print("\n\n Please use one of the following keras applications only [ \"vgg16\", \"vgg19\", \"resnet50\", \"xception\", \"inceptionv3\", \"inceptionresnetv2\", \"densenet\", \"mobilenetv2\" ] or False")
				sys.exit()
			pred = model1.predict(self.images)
			print('Finished Predicting')
			images_temp = pred.reshape(self.images.shape[0], -1)
			if self.use_pca == False: 
				self.images_new = images_temp
				print('not using pca') 
			else:
				print('using pca') 
				model2 = PCA(n_components=None, random_state=728)
				model2.fit(images_temp)
				self.images_new = model2

	def clustering(self):
		model = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=728)
		model.fit(self.images_new)
		print('first clustering complete, generating images')

		#temp.compute_best()
		
		print('printed image')
		predictions = model.predict(self.images_new)
		#print(predictions)
		for i in range(self.max_examples):
			shutil.copy2(self.folder_path+self.image_paths[i], "output/cluster"+str(predictions[i]))
		print("\n Clustering complete! \n\n Clusters and the respective images are stored in the \"output\" folder.")

	
	def compute_best(self):
		mms = MinMaxScaler()
		mms.fit(self.images_new)
		data_transformed = mms.transform(self.images_new)

		Sum_of_squared_distances = []
		K = range(1,26)
		for k in K:
			km = KMeans(n_clusters=k, n_jobs=-1, random_state=728)
			km = km.fit(data_transformed)
			Sum_of_squared_distances.append(km.inertia_)
			print("K %d processed" % k)
		
		plt.plot(K, Sum_of_squared_distances, 'bx-')
		plt.xlabel('k')
		plt.ylim(0, 5e6)
		plt.ylabel('Sum_of_squared_distances')
		plt.title('Elbow Method For Optimal k')
		plt.show()

	def make_stitch(self):
		print("Making stitch...")
		for i in range(number_of_clusters):
			temp_directory = "output/cluster"+str(i)+"/*png"
			temp_filelist = glob.glob(temp_directory)

			length = float(len(temp_filelist))

			
			probability = (length/10.)/100.
			images = []

			for file_name in temp_filelist:
				decision = np.random.choice([0,1], p=[1.-probability, probability])
				if decision == 1:
					images.append(cv2.cvtColor(cv2.resize(cv2.imread(file_name), (256,256)), cv2.COLOR_BGR2RGB))
					print(file_name, " appended!")
				else:
					print(file_name, " skipped!")

			fig=plt.figure(figsize=(8, 8))
			columns = 3
			rows = 3
			for x in range(1, columns*rows +1):
				if x >= len(images):
					files = os.listdir("output/cluster"+str(i)+"/")
					index = random.randrange(0, len(files))
					print(files[index], "was chosen randomly!")
					img = cv2.cvtColor(cv2.resize(cv2.imread("output/cluster"+str(i)+"/"+files[index]), (256,256)), cv2.COLOR_BGR2RGB)
				else:
					img = images[x]
				fig.add_subplot(rows, columns, x)
				plt.imshow(img)
			plt.savefig("output/stitches/concat"+str(i)+".png")


			del temp_directory
			del temp_filelist
			del images


if __name__ == "__main__":

	print("\n\n \t\t START\n\n")

	number_of_clusters = 10 # cluster names will be 0 to number_of_clusters-1

	data_path = "../data/images/full_pngs/" # path of the folder that contains the images to be considered for the clustering (The folder must contain only image files)

	max_examples = None # number of examples to use, if "None" all of the images will be taken into consideration for the clustering
	# If the value is greater than the number of images present  in the "data_path" folder, it will use all the images and change the value of this variable to the number of images available in the "data_path" folder. 

	use_imagenets = 'DenseNet'
	# choose from: "Xception", "VGG16", "VGG19", "ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet", "MobileNetV2" and "False" -> Default is: False
	# you have to use the correct spelling! (case of the letters are irrelevant as the lower() function has been used)

	if use_imagenets == False:
		use_pca = False
	else:
		use_pca = False # Make it True if you want to use PCA for dimentionaity reduction -> Default is: False

	temp = image_clustering(data_path, number_of_clusters, max_examples, use_imagenets, use_pca)
	temp.load_images()
	temp.get_new_imagevectors()
	temp.clustering()
	temp.make_stitch()

	print("\n\n\t\t END\n\n")