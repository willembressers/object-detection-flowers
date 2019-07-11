# https://giusedroid.blogspot.com/2015/04/using-python-and-k-means-in-hsv-color.html
# https://charlesleifer.com/blog/using-python-and-k-means-to-find-the-dominant-colors-in-images/
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097




import numpy as np
import cv2

img = cv2.imread('sunflower.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

colors = ['rgb({red}, {green}, {blue})'.format(red=int(color[2]), green=int(color[1]), blue=int(color[0])) for color in center]
print(colors)
# unique, counts = np.unique(label, return_counts=True)

# data = {'count':[], 'color':[]}
# for cluster_id in unique:
# 	data['count'].append(counts[cluster_id])
# 	data['color'].append('rgb({red}, {green}, {blue})'.format(red=int(center[cluster_id][2]), green=int(center[cluster_id][1]), blue=int(center[cluster_id][0])))
# print(data)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()





# import cv2
# import time as t
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.cluster.vq import vq, kmeans
# from sklearn.cluster import KMeans

# def find_histogram(clt):
#     """
#     create a histogram with k clusters
#     :param: clt
#     :return:hist
#     """
#     numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
#     (hist, _) = np.histogram(clt.labels_, bins=numLabels)

#     hist = hist.astype("float")
#     hist /= hist.sum()

#     return hist

# def plot_colors2(hist, centroids):
#     bar = np.zeros((50, 300, 3), dtype="uint8")
#     startX = 0

#     for (percent, color) in zip(hist, centroids):
#         # plot the relative percentage of each cluster
#         endX = startX + (percent * 300)
#         cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
#                       color.astype("uint8").tolist(), -1)
#         startX = endX

#     # return the bar chart
#     return bar

# img = cv2.imread('sunflower.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
# clt = KMeans(n_clusters=3) #cluster number
# clt.fit(img)

# hist = find_histogram(clt)
# bar = plot_colors2(hist, clt.cluster_centers_)

# plt.axis("off")
# plt.imshow(bar)
# plt.show()






# img = cv2.imread('sunflower.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
# plt.figure(figsize=(10,8))
# plt.subplot(311)                             #plot in the first cell
# plt.subplots_adjust(hspace=.5)
# plt.title("Hue")
# plt.hist(np.ndarray.flatten(hue), bins=180)
# plt.subplot(312)                             #plot in the second cell
# plt.title("Saturation")
# plt.hist(np.ndarray.flatten(sat), bins=128)
# plt.subplot(313)                             #plot in the third cell
# plt.title("Luminosity Value")
# plt.hist(np.ndarray.flatten(val), bins=128)
# plt.show()










# def do_cluster(hsv_image, K, channels):
#     # gets height, width and the number of channes from the image shape
#     h,w,c = hsv_image.shape
#     # prepares data for clustering by reshaping the image matrix into a (h*w) x c matrix of pixels
#     cluster_data = hsv_image.reshape( (h*w,c) )
#     # grabs the initial time
#     t0 = t.time()
#     # performs clustering
#     codebook, distortion = kmeans(np.array(cluster_data[:,0:channels], dtype=np.float), K)
#     # takes the final time
#     t1 = t.time()
#     print("Clusterization took %0.5f seconds" % (t1-t0))
    
    
#     # calculates the total amount of pixels
#     # tot_pixels = h*w
#     # # generates clusters
#     # data, dist = vq(cluster_data[:,0:channels], codebook)
#     # # calculates the number of elements for each cluster
#     # weights = [len(data[data == i]) for i in range(0,K)]
    
#     # # creates a 4 column matrix in which the first element is the weight and the other three
#     # # represent the h, s and v values for each cluster
#     # color_rank = np.column_stack((weights, codebook))
#     # # sorts by cluster weight
#     # color_rank = color_rank[np.argsort(color_rank[:,0])]

#     # # creates a new blank image
#     # new_image =  np.array([0,0,255], dtype=np.uint8) * np.ones( (500, 500, 3), dtype=np.uint8)
#     # img_height = new_image.shape[0]
#     # img_width  = new_image.shape[1]

#     # # for each cluster
#     # for i,c in enumerate(color_rank[::-1]):
        
#     #     # gets the weight of the cluster
#     #     weight = c[0]
        
#     #     # calculates the height and width of the bins
#     #     height = int(weight/float(tot_pixels) *img_height )
#     #     width = img_width/len(color_rank)

#     #     # calculates the position of the bin
#     #     x_pos = i*width


        
#     #     # defines a color so that if less than three channels have been used
#     #     # for clustering, the color has average saturation and luminosity value
#     #     color = np.array( [0,128,200], dtype=np.uint8)
        
#     #     # substitutes the known HSV components in the default color
#     #     for j in range(len(c[1:])):
#     #         color[j] = c[j+1]
        
#     #     # draws the bin to the image
#     #     new_image[ img_height-height:img_height, x_pos:x_pos+width] = [color[0], color[1], color[2]]
        
#     # # returns the cluster representation
#     # return new_image

# for i in range(1,4):
#     plt.subplot(141 + i)
#     plt.title("Channels: %i" % i)
#     new_image = do_cluster(hsv, 5, i)
#     # new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
#     # plt.imshow(new_image)

# # hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

# # plt.imshow(hist, interpolation = 'nearest')
# # plt.show()

# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()