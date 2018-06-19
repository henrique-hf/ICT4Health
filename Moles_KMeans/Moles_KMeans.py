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

colors = []
for i in centroids:
    colors.append(i.sum())
darkest_color = colors.index(min(colors))

plt.figure()
plt.imshow(im)

plt.figure()
plt.imshow(im_3D)

#%% find center
middle_N1 = int(N1/2)
middle_N2 = int(N2/2)

count_h = np.count_nonzero(im_3D[middle_N1,:] == darkest_color)

temp = 0
for i in range(N2):
    if im_3D[middle_N1,i] == darkest_color:
        temp += 1
    if temp == int(count_h/2):
        center_h = i
        break
    
count_v = np.count_nonzero(im_3D[:,middle_N2] == darkest_color)
        
temp = 0
for i in range(N1):
    if im_3D[i,middle_N2] == darkest_color:
        temp += 1
    if temp == (count_v/2):
        center_v = i
        break
    
#%% crop
new_im_3D = im_3D
i = center_h
while i >= 0:
    pixel_y = np.count_nonzero(new_im_3D[i,:] == darkest_color)
    if pixel_y > 0:
        i -= 1
    elif pixel_y == 0:
        crop_sup = i+1
        break

i = center_h
while i < N1:
    pixel_y = np.count_nonzero(new_im_3D[i,:] == darkest_color)
    if pixel_y > 0:
        i += 1
    elif pixel_y == 0:
        crop_inf = i-1
        break
    
i = center_v
while i >= 0:
    pixel_y = np.count_nonzero(new_im_3D[:,i] == darkest_color)
    if pixel_y > 0:
        i -= 1
    elif pixel_y == 0:
        crop_left = i+1
        break

i = center_v
while i < N2:
    pixel_y = np.count_nonzero(new_im_3D[:,i] == darkest_color)
    if pixel_y > 0:
        i += 1
    elif pixel_y == 0:
        crop_right = i-1
        break
    

new_im_3D = new_im_3D[crop_sup:crop_inf, crop_left:crop_right]
plt.figure()
plt.imshow(new_im_3D)

#%% contour columns
min_col = []
max_col = []
[M1, M2] = new_im_3D.shape
for i  in range(M2):
    for j in range(M1):
        if new_im_3D[j,i] == darkest_color:
            min_col.append((j,i))
            break

for i in range(M2):
    for j in reversed(range(M1)):
        if new_im_3D[j,i] == darkest_color:
            max_col.append((j,i))
            break

#%% contour rows
min_row = []
max_row = []
for i in range(M1):
    for j in range(M2):
        if new_im_3D[i,j] == darkest_color:
            min_row.append((i,j))
            break
        
for i in range(M1):
    for j in reversed(range(M2)):
        if new_im_3D[i,j] == darkest_color:
            max_row.append((i,j))
            break
        
#%% plot contour
contour = np.zeros((M1,M2))
list_contour = min_col + max_col + min_row + max_row
for i in list_contour:
    contour[i] = 1

plt.matshow(contour)

#%% plot
plt.show()
