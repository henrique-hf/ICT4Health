import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

seed = 10

file_path = 'Data\low_risk_1.jpg'

im = mpimg.imread(file_path)

[N1, N2, N3] = im.shape
im_2D = im.reshape((N1*N2, N3))

kmeans = KMeans(n_clusters=3, random_state=seed)
kmeans.fit(im_2D)

centroids = kmeans.cluster_centers_.astype('uint8')
labels = kmeans.labels_

im_3D = kmeans.predict(im_2D).reshape((N1, N2))

plt.imshow(im_3D)
plt.show()
