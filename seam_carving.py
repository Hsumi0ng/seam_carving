
#copy from jupyter
from __future__ import print_function

# Setup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import color
from time import time
from IPython.display import HTML

%matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 12.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
%load_ext autoreload              # for auto-reloading extenrnal modules
%autoreload 2

# Load image
from skimage import io, util
img = io.imread('imgs/broadway_tower.jpg')
img = util.img_as_float(img)

plt.title('Original Image')
plt.imshow(img)
plt.show()

# Compute energy function
from seam_carving import energy_function
start = time()
energy = energy_function(img)
end = time()
print("Computing energy function: %f seconds." % (end - start))
plt.title('Energy')
plt.axis('off')
plt.imshow(energy)
plt.show()

# Vertical Cost Map
from seam_carving import compute_cost
start = time()
vcost, _ = compute_cost(_, energy, axis=1)  # don't need the first argument for compute_cost
end = time()

print("Computing vertical cost map: %f seconds." % (end - start))

plt.title('Vertical Cost Map')
plt.axis('off')
plt.imshow(vcost, cmap='inferno')
plt.show()

# Horizontal Cost Map
start = time()
hcost, _ = compute_cost(_, energy, axis=0)
end = time()
print("Computing horizontal cost map: %f seconds." % (end - start))
plt.title('Horizontal Cost Map')
plt.axis('off')
plt.imshow(hcost, cmap='inferno')
plt.show()

from seam_carving import backtrack_seam
vcost, vpaths = compute_cost(img, energy)

# Vertical Backtracking
start = time()
end = np.argmin(vcost[-1])
seam_energy = vcost[-1, end]
seam_ = backtrack_seam(vpaths, end)
end = time()

print("Backtracking optimal seam: %f seconds." % (end - start))
print('Seam Energy:', seam_energy)

# Visualize seam
vseam = np.copy(img)
for row in range(vseam.shape[0]):
    vseam[row, seam_[row], :] = np.array([1.0, 0, 0])
plt.title('Vertical Seam')
plt.axis('off')
plt.imshow(vseam)
plt.show()

# Reduce image width
from seam_carving import reduce
H, W, _ = img.shape
W_new = 400
start = time()
out = reduce(img, W_new)
end = time()
print("Reducing width from %d to %d: %f seconds." % (W, W_new, end - start))
plt.subplot(2, 1, 1)
plt.title('Original')
plt.imshow(img)
plt.subplot(2, 1, 2)
plt.title('Resized')
plt.imshow(out)
plt.show()

# Reduce image height
H, W, _ = img.shape
H_new = 300
start = time()
out = reduce(img, H_new, axis=0)
end = time()
print("Reducing height from %d to %d: %f seconds." % (H, H_new, end - start))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title('Resized')
plt.imshow(out)
plt.show()

# This is a naive implementation of image enlarging
from seam_carving import enlarge_naive
W_new = 800
start = time()
enlarged = enlarge_naive(img, W_new)
end = time()
print("Enlarging(naive) width from %d to %d: %f seconds." \
      % (W, W_new, end - start))

plt.imshow(enlarged)
plt.show()

# Alternatively, find k seams for removal and duplicate them.
from seam_carving import find_seams
start = time()
seams = find_seams(img, W_new - W)
end = time()
print("Finding %d seams: %f seconds." % (W_new - W, end - start))
plt.imshow(seams, cmap='viridis')
plt.show()

from seam_carving import enlarge
W_new = 800
start = time()
out = enlarge(img, W_new)
end = time()
print("Enlarging width from %d to %d: %f seconds." \
      % (W, W_new, end - start))
plt.subplot(2, 1, 1)
plt.title('Original')
plt.imshow(img)
plt.subplot(2, 1, 2)
plt.title('Resized')
plt.imshow(out)
plt.show()

#Map of the seams for horizontal seams.
start = time()
seams = find_seams(img, W_new - W, axis=0)
end = time()
print("Finding %d seams: %f seconds." % (W_new - W, end - start))
plt.imshow(seams, cmap='viridis')
plt.show()


H_new = 600

start = time()
out = enlarge(img, H_new, axis=0)
end = time()
print("Enlarging height from %d to %d: %f seconds."% (H, H_new, end - start))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title('Resized')
plt.imshow(out)
plt.show()









