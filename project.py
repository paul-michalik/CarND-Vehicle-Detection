import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

class FeatureExtractionArgs:
    colorspace = None
    orient = None
    pix_per_cell = None
    cell_per_block = None
    hog_channel = None
    spatial_size = None
    hist_bins = None  
    spatial_feat = None 
    hist_feat = None 
    hog_feat = None

    def __init__(self,
                 colorspace = 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                 orient = 11,
                 pix_per_cell = 16,
                 cell_per_block = 2,
                 hog_channel = 'ALL', # Can be 0, 1, 2, or "ALL"
                 spatial_size=(32, 32),
                 hist_bins=32,
                 spatial_feat=False, 
                 hist_feat=False, 
                 hog_feat=True):
        self.colorspace = colorspace
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins=hist_bins  
        self.spatial_feat = spatial_feat 
        self.hist_feat = hist_feat 
        self.hog_feat = hog_feat

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True, transform_sqrt=False):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=transform_sqrt, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=transform_sqrt, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, 
                     color_space='RGB', 
                     orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_size=(32, 32),
                     hist_bins=32,   
                     spatial_feat=False, 
                     hist_feat=False, 
                     hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def extract_features_shuffle_and_split(car_images, noncar_images, args):
    # Feature extraction parameters
    colorspace = args.colorspace
    orient = args.orient
    pix_per_cell = args.pix_per_cell#
    cell_per_block = args.cell_per_block
    hog_channel = args.hog_channel

    t = time.time()
    car_features = extract_features(car_images, 
                                    color_space=colorspace, 
                                    orient=orient, 
                                    pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel,
                                    spatial_feat=False,
                                    hist_feat=False)
    notcar_features = extract_features(noncar_images,
                                       color_space=colorspace, 
                                       orient=orient, 
                                       pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel,
                                       spatial_feat=False,
                                       hist_feat=False)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)  

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    return X_train, X_test, y_train, y_test

def train_classifier(X_train, X_test, y_train, y_test, n_predict=10):
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    return svc

# Tests. No logic beyond this point.

def test_get_hog_features(car_img, noncar_img):
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    _, car_dst = get_hog_features(car_img[:,:,2], 
                                  orient=orient, 
                                  pix_per_cell=pix_per_cell, 
                                  cell_per_block=cell_per_block, 
                                  vis=True, 
                                  feature_vec=True)
    _, noncar_dst = get_hog_features(noncar_img[:,:,2],  
                                  orient=orient, 
                                  pix_per_cell=pix_per_cell, 
                                  cell_per_block=cell_per_block, 
                                  vis=True, 
                                  feature_vec=True)

    # Visualize 
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,14))
    f.subplots_adjust(hspace = .4, wspace=.2)
    ax1.imshow(car_img)
    ax1.set_title('Car Image', fontsize=16)
    ax2.imshow(car_dst, cmap='gray')
    ax2.set_title('Car HOG', fontsize=16)
    ax3.imshow(noncar_img)
    ax3.set_title('Non-Car Image', fontsize=16)
    ax4.imshow(noncar_dst, cmap='gray')
    ax4.set_title('Non-Car HOG', fontsize=16)

def test_extract_features_shuffle_and_split(car_images, noncar_images, args):
    X_train, X_test, y_train, y_test = extract_features_shuffle_and_split(car_images, noncar_images, args)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

def test_train_classifier(cars, notcars, args, n_predict=10):
    X_train, X_test, y_train, y_test = extract_features_shuffle_and_split(cars, 
                                                                          notcars, 
                                                                          args)

    t = time.time()
    svc = train_classifier(X_train, X_test, y_train, y_test, n_predict)
    print(round(time.time()-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    print('Predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    print(round(time.time()-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

if __name__ == '__main__':
    cars = glob.glob('vehicles/*/*.png')
    notcars = glob.glob('non-vehicles/*/*.png')
    
    #test_get_hog_features(mpimg.imread(cars[5]), mpimg.imread(notcars[5]))
    
    ## Feature extraction parameters 1
    #args = FeatureExtractionArgs(colorspace = 'LUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #                             orient = 9,
    #                             pix_per_cell = 16,
    #                             cell_per_block = 2,
    #                             hog_channel = 'ALL') # Can be 0, 1, 2, or "ALL"
    #test_extract_features_shuffle_and_split(cars[0:min(500, len(cars))], 
    #                                        notcars[0:min(500, len(notcars))], 
    #                                        args)

    ## Feature extraction parameters 2  
    #args = FeatureExtractionArgs(colorspace = 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #                             orient = 11,
    #                             pix_per_cell = 16,
    #                             cell_per_block = 2,
    #                             hog_channel = 'ALL') # Can be 0, 1, 2, or "ALL"
    #test_extract_features_shuffle_and_split(cars[0:100], notcars[0:100], args)

    args = FeatureExtractionArgs(colorspace = 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                                 orient = 11,
                                 pix_per_cell = 16,
                                 cell_per_block = 2,
                                 hog_channel = 'ALL') # Can be 0, 1, 2, or "ALL"    
    #test_train_classifier(cars[0:min(1000, len(cars))],
    #                      notcars[0:min(1000, len(notcars))],
    #                      args,
    #                      n_predict=100)

    test_train_classifier(cars,
                          notcars,
                          args,
                          n_predict=100)
