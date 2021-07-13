import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

def concat(filename):

    cluster_img = cv2.imread(cluster_prefix+filename)
    lrp_img = cv2.imread(lrp_prefix+filename)

    
    im_h = cv2.hconcat([cluster_img, lrp_img])
  
    #show the output image
    cv2.imwrite(concat_prefix+filename, im_h)

cluster_directory = '../clustering/KTLX_4/cropped_clusters/cluster3/*png*'
cluster_list = glob.glob(cluster_directory)
lrp_prefix = '../clustering/KTLX_4/cropped_clusters/LRP_results/'
cluster_prefix = '../clustering/KTLX_4/cropped_clusters/'
concat_prefix = '../clustering/KTLX_4/cropped_clusters/LRP_concat/'
i = 0

def make_stitch():
		print("Making stitch...")
		for i in range(4):
			temp_directory = "../clustering/KTLX_4/cropped_clusters/LRP_concat/cluster"+str(i)+"/*png"
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
					files = os.listdir("../clustering/KTLX_4/cropped_clusters/LRP_concat/cluster"+str(i)+"/")
					index = random.randrange(0, len(files))
					print(files[index], "was chosen randomly!")
					img = cv2.cvtColor(cv2.resize(cv2.imread("../clustering/KTLX_4/cropped_clusters/LRP_concat/cluster"+str(i)+"/"+files[index]), (512,512)), cv2.COLOR_BGR2RGB)
				else:
					img = images[x]
				fig.add_subplot(rows, columns, x)
				plt.imshow(img)
			plt.savefig("../clustering/KTLX_4/cropped_clusters/LRP_concat/stitches/concat"+str(i)+".png")


			del temp_directory
			del temp_filelist
			del images


if __name__ == "__main__":
    make_stitch()
    """
    for filename in cluster_list:
        concat(filename[38:])
        i+=1
        if i % 10 == 0:
            print("File %d of %d processed" % (i, len(cluster_list)))
    
    Cluster = SLURMCluster(processes=6, cores=36, memory='128GB', walltime='2:00:00')
    Cluster.scale(36)
    client = Client(Cluster)
    print("Waiting for workers...")
    while(len(client.scheduler_info()["workers"]) < 6):
        i = 1
    futures = client.map(make_LRP, data_list)
    wait(futures)
    client.close() """