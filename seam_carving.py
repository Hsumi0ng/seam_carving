import numpy as np
from   skimage import color

def energy_function(image):
    
    #For each pixel, we will sum the absolute value of the gradient in each direction.
    H, W, _ = image.shape
    out = np.zeros((H, W))
    gray_image = color.rgb2gray(image)

    ### YOUR CODE HERE
    out = np.abs(np.gradient(gray_image)[0]) + np.abs(np.gradient(gray_image)[1])
    pass
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    
    #Computes optimal cost map (vertical) and paths of the seams.
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    stackmat = np.zeros((3, W))
    stackmat[0, 0] = W
    stackmat[2, W-1] = W
    index = list( map( int , np.linspace(0, W-1, W) ) )
    for i in range(1, H):
        minway = np.argsort(np.argsort(cost[i-1]))
        stackmat[0, 1:W] =minway[0:W-1]
        stackmat[1]=minway
        stackmat[2, 0:W-1] = minway[1:W]
        paths[i] = np.argmin(stackmat,axis=0)-1
        costi_1 = cost[i-1]
        cost[i] = costi_1[index+paths[i]]+energy[i]
    pass
    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    
    #Backtracks the paths map to find the seam ending at (H-1, end)
    H, W = paths.shape
    # initialize with -1 to make sure that everything gets modified
    seam = - np.ones(H, dtype=np.int)

    # Initialization
    seam[H-1] = end

    ### YOUR CODE HERE
    listmap = list( map( int , np.linspace(0, W-1, W) ) )
    pathslistmap = paths+np.tile(listmap,H).reshape(-1,W)
    for i in range(2,H+1):
        # pathsh_i = paths[H-i+1]+listmap
        # seam[H-i] = pathsh_i[seam[H-i+1]]
        seam[H-i] = pathslistmap[H-i+1, seam[H-i+1]]
    pass
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    
    #Remove a seam from the image.
    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE
    out = np.zeros((H, W-1, C))
    for i in range(H):
        # imagei = image[i]
        out[i] = np.delete(image[i],seam[i],0)
    if image.dtype == int:
        out = out.astype(np.int32)
    pass
    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1
    # Make sure that `out` has same type as `image`
    assert out.dtype == image.dtype, \
       "Type changed between image (%s) and out (%s) in remove_seam" % (image.dtype, out.dtype)

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost, bfunc=backtrack_seam, rfunc=remove_seam):
    
    #Reduces the size of the image using the seam carving process.
    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    for i in range(W-size):
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = bfunc(paths, end)
        out = rfunc(out, seam)
    pass      
    ### END YOUR CODE
    
    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    
    #Duplicates pixels of the seam, making the pixels on the seam path "twice larger".
    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    for i in range(H):
        # np.insert(arr,[1],arr2,axis=1)
        out[i] = np.insert(image[i], seam[i], [image[i,seam[i]]], axis=0)
    pass
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost, bfunc=backtrack_seam, dfunc=duplicate_seam):
    
    #Increases the size of the image using the seam duplication process.
    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    for i in range(size-W):
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = bfunc(paths, end)
        out = dfunc(out, seam)
    pass
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost, bfunc=backtrack_seam, rfunc=remove_seam):
    
    #Find the top k seams (with lowest energy) in the image.
    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W
    indices = np.tile(range(W), (H, 1))  # shape (H, W)
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = bfunc(paths, end)

        # Remove that seam from the image
        image = rfunc(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = rfunc(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost, dfunc=duplicate_seam, bfunc=backtrack_seam, rfunc=remove_seam):
   
    #Enlarges the size of the image by duplicating the low energy seams.
    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE
    seams = find_seams(out, size - W)
    index = np.zeros((H, size),dtype = int)
    for i in range(size-W):
        seam = np.where(seams == i+1)[1]
        seamenlarge = index[:,0:W+i+1]
        for j in range(H):
            seamenlarge[j] = np.insert(seams[j], seam[j], i+1, axis=0)
        seams = seamenlarge
        out = dfunc(out, seam)
    pass 
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))
        
    return out
