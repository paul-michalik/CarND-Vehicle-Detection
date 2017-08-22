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
    use_spatial_feat = None 
    use_hist_feat = None 
    use_hog_feat = None

    def __init__(self,
                 colorspace = 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                 orient = 11,
                 pix_per_cell = 16,
                 cell_per_block = 2,
                 hog_channel = 'ALL', # Can be 0, 1, 2, or "ALL"
                 spatial_size=(32, 32),
                 hist_bins=32,
                 use_spatial_feat=False, 
                 use_hist_feat=False, 
                 use_hog_feat=True):
        self.colorspace = colorspace
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins=hist_bins  
        self.use_spatial_feat = use_spatial_feat 
        self.use_hist_feat = use_hist_feat 
        self.use_hog_feat = use_hog_feat

class FeatureExtraction:
    args = FeatureExtractionArgs()
    car_features = None
    noncar_features = None
    X_train = None 
    X_test = None
    y_train = None
    y_test = None
    X_scaler = None
    classifier = None

    def __init__(self, args = FeatureExtractionArgs()):
        self.args = args

    def extract_feature_category(self, image_files):
        return extract_features(imgs=image_files, 
                                color_space=self.args.colorspace,
                                orient=self.args.orient,
                                pix_per_cell=self.args.pix_per_cell,
                                cell_per_block=self.args.cell_per_block,
                                hog_channel=self.args.hog_channel,
                                spatial_size=self.args.spatial_size,
                                hist_bins=self.args.hist_bins,
                                spatial_feat=self.args.use_spatial_feat,
                                hist_feat=self.args.use_hist_feat,
                                hog_feat=self.args.use_hog_feat)

    def extract_features(self, car_image_files, notcar_image_files):
        car_features = self.extract_feature_category(car_image_files)
        noncar_features = self.extract_feature_category(notcar_image_files)
        self.car_features, self.noncar_features = car_features, noncar_features


    def save_features(filename):
        return None

    def load_features(filename):
        return None

    def prepare_features(self, scale=True):
        # Create an array stack of feature vectors
        X = np.vstack((self.car_features, self.noncar_features)).astype(np.float64) 

        if scale == True:
            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)
        else:
            scaled_X = X

        # Define the labels vector
        y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.noncar_features))))

        # Split up data into randomized training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=0)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
                
        if scale == True:
            self.X_scaler = X_scaler

class FeatureClassification:
    classifier = None
    features = FeatureExtraction()
    boxes = []

    def __init__(self, features = None):
        self.features = features

    def train_classifier(self, classifier='LinearSVC'):
        if classifier in 'LinearSVC':
            svc = LinearSVC().fit(self.features.X_train, self.features.y_train)
            self.classifier = svc
        # add other alternatives

    def find_cars(self, image, ystart, ystop, scale, xstart=0, classifier='LinearSVC', show_all_rectangles=False):
        cspace=self.features.args.colorspace
        hog_channel=self.features.args.hog_channel
        svc = self.classifier
        X_scaler = self.features.X_scaler
        orient=self.features.args.orient
        pix_per_cell=self.features.args.pix_per_cell
        cell_per_block=self.features.args.cell_per_block
        spatial_size=self.features.args.spatial_size
        hist_bins=self.features.args.hist_bins

        return find_cars(img=image, xstart=xstart, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size,hist_bins=hist_bins, show_all_rectangles=show_all_rectangles)

    def find_cars_and_store_boxes(self, image, ystart, ystop, scale, classifier='LinearSVC', show_all_rectangles=False):
        self.boxes.extend(self.find_cars(image, ystart, ystop, scale, classifier, show_all_rectangles))

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


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, xstart=0, show_all_rectangles=False):

    # array of rectangles where cars were detected
    rectangles = []
    
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(image)   
    
    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # select colorspace channel for HOG 
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)   
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    #print(xstart, nxsteps)
    for xb in range(xstart, nxsteps):
        #print(xb)
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            if X_scaler != None:
                test_features = X_scaler.transform(np.hstack(hog_features).reshape(1, -1)) 
                test_prediction = svc.predict(test_features)
            else:
                test_prediction = svc.predict(hog_features)
            
            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                yield ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                
    #return rectangles

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Tests. No logic beyond this point.

def test_get_hog_features(car_img, noncar_img, figsize=(7,7), fontsize=16):
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
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    f.subplots_adjust(hspace = .4, wspace=.2)
    ax1.imshow(car_img)
    ax1.set_title('Car Image', fontsize=16)
    ax2.imshow(car_dst, cmap='gray')
    ax2.set_title('Car HOG', fontsize=16)
    ax3.imshow(noncar_img)
    ax3.set_title('Non-Car Image', fontsize=16)
    ax4.imshow(noncar_dst, cmap='gray')
    ax4.set_title('Non-Car HOG', fontsize=16)
    plt.show()

def test_FeatureExtraction_extract_feature_category(car_images, noncar_images, args = FeatureExtractionArgs()):

    t = time.time();
    feat_ext = FeatureExtraction(args)
    car_features = feat_ext.extract_feature_category(car_images)
    noncar_features = feat_ext.extract_feature_category(noncar_images)

    print('Using {} orientations, {} pixels per cell and {} cells per block'.format(feat_ext.args.orient,
                                                                                    feat_ext.args.pix_per_cell,
                                                                                    feat_ext.args.cell_per_block))

    print('Feature vector length cars = {}, non cars = {}'.format(len(car_features), 
                                                                  len(noncar_features)))

    print('time spent: {:.2f} s'.format(time.time() - t))

def test_FeatureExtraction_extract_features(car_images, noncar_images, args = FeatureExtractionArgs()):

    t = time.time();
    feat_ext = FeatureExtraction(args)
    feat_ext.extract_features(car_images, noncar_images)

    print('Using {} orientations, {} pixels per cell and {} cells per block'.format(feat_ext.args.orient,
                                                                                    feat_ext.args.pix_per_cell,
                                                                                    feat_ext.args.cell_per_block))

    print('Feature vector length cars = {}, non cars = {}'.format(len(feat_ext.car_features), 
                                                                  len(feat_ext.noncar_features)))

    print('time spent: {:.2f} s'.format(time.time() - t))

def test_FeatureExtraction_prepare_features(car_images, noncar_images, args = FeatureExtractionArgs(), scale = True):
    t = time.time();

    feat_ext = FeatureExtraction(args)
    feat_ext.extract_features(car_images, noncar_images)
    feat_ext.prepare_features(scale = scale)

    print('Using {} orientations, {} pixels per cell and {} cells per block'.format(feat_ext.args.orient,
                                                                                    feat_ext.args.pix_per_cell,
                                                                                    feat_ext.args.cell_per_block))

    print('shapes: X_train, X_test, y_train, y_test'.format(np.shape(feat_ext.X_train),
                                                    np.shape(feat_ext.X_test),
                                                    np.shape(feat_ext.y_train),
                                                    np.shape(feat_ext.y_test)))

    print('time spent: {:.2f} s'.format(time.time() - t))

def test_FeatureClassification_train_classifier(feat_ext = FeatureExtraction(), n_predict=10):
    X_test, y_test = feat_ext.X_test, feat_ext.y_test

    t = time.time();
    feat_class = FeatureClassification(feat_ext)
    feat_class.train_classifier('LinearSVC')
    print('time spent for training: {:.2f} s'.format(time.time() - t))

    # Check the score of the SVC
    print('Test Accuracy = {:.2f}'.format(feat_class.classifier.score(X_test, y_test)))

    # Check the prediction time for a single sample
    t=time.time()
    print('predicted labels:\n', feat_class.classifier.predict(X_test[0:n_predict]))
    print('test labels     :\n', y_test[0:n_predict])
    print('time spent for prediction: {:.2f} s'.format(time.time() - t))

def test_FeatureClassification_with_trained_classifier(feat_class = FeatureClassification(), n_predict=10):
     
    X_test, y_test = feat_class.features.X_test, feat_class.features.y_test

    # Check the score of the SVC
    print('Test Accuracy = {:.2f}'.format(feat_class.classifier.score(X_test, y_test)))

    # Check the prediction time for a single sample
    print('predicted labels:\n', feat_class.classifier.predict(X_test[0:n_predict]))
    print('test labels     :\n', y_test[0:n_predict])

def test_FeatureClassification_find_cars_single_image(test_image, 
                                                      feat_class = FeatureClassification(), 
                                                      ystart = 400, 
                                                      ystop = 656, 
                                                      scale = 1.5):
    rectangles = list(feat_class.find_cars(test_image, ystart=ystart, ystop=ystop, scale=scale))
    print(len(rectangles), 'rectangles found in image')

def test_FeatureClassification_find_cars_and_draw_boxes_single_image(test_image, 
                                                      feat_class = FeatureClassification(), 
                                                      ystart = 400, 
                                                      ystop = 656, 
                                                      scale = 1.5):
    rectangles = list(feat_class.find_cars(test_image, ystart=ystart, ystop=ystop, scale=scale))
    print(len(rectangles), 'rectangles found in image')
    test_img_rects = draw_boxes(test_image, rectangles)
    plt.figure(figsize=(10,10))
    plt.imshow(test_img_rects)
    plt.show()

def test_FeatureClassification_find_cars_multi_scale_single_image(test_image, 
                                                                  feat_class = FeatureClassification(), 
                                                                  xstart=0):

    rectangles = []
    for ystart, ystop, scale in [(400, 464, 1.0), (416, 480, 1.0), (400, 496, 1.5), (432, 528, 1.5), (400, 528, 2.0), (432, 560, 2.0), (400, 596, 3.5), (464, 660, 3.5)]:
        cur_rectangles = list(feat_class.find_cars(test_image, xstart=xstart, ystart=ystart, ystop=ystop, scale=scale))
        print('{} rectangles found in image for ystart = {}, ystop = {}, scale = {}'.format(len(cur_rectangles),
                                                                                            ystart,
                                                                                            ystop, 
                                                                                            scale))
        rectangles.extend(cur_rectangles)

    test_img_rects = draw_boxes(test_image, rectangles, color=(0, 0, 255), thick=2)
    plt.figure(figsize=(10,10))
    plt.imshow(test_img_rects)

if __name__ == '__main__':
    cars = glob.glob('vehicles/*/*.png')
    notcars = glob.glob('non-vehicles/*/*.png')
    
    n_features_max = min(100, min(len(cars), len(notcars)))
    n_predict_max = min(50, n_features_max/2)

    #test_get_hog_features(mpimg.imread(cars[5]), mpimg.imread(notcars[5]))   

    ## Feature extraction parameters 1
    #args = FeatureExtractionArgs(colorspace = 'LUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #                             orient = 9,
    #                             pix_per_cell = 16,
    #                             cell_per_block = 2,
    #                             hog_channel = 'ALL') # Can be 0, 1, 2, or "ALL"

    #test_FeatureExtraction_extract_feature_category(cars[0:n_features_max], 
    #                                                notcars[0:n_features_max],
    #                                                args)
    #test_FeatureExtraction_extract_features(cars[0:n_features_max], 
    #                                        notcars[0:n_features_max],
    #                                        args)

    #test_FeatureExtraction_prepare_features(cars[0:n_features_max], 
    #                                        notcars[0:n_features_max],
    #                                        args,
    #                                        scale=False)


    ## Feature extraction parameters 2  
    #args = FeatureExtractionArgs(colorspace = 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #                             orient = 11,
    #                             pix_per_cell = 16,
    #                             cell_per_block = 2,
    #                             hog_channel = 'ALL') # Can be 0, 1, 2, or "ALL"
    
    #test_FeatureExtraction_extract_feature_category(cars[0:n_features_max], 
    #                                                notcars[0:n_features_max],
    #                                                args)
    #test_FeatureExtraction_extract_features(cars[0:n_features_max], 
    #                                        notcars[0:n_features_max],
    #                                        args)

    #test_FeatureExtraction_prepare_features(cars[0:n_features_max], 
    #                                        notcars[0:n_features_max],
    #                                        args,
    #                                        scale=True)

    feat_ext = FeatureExtraction(FeatureExtractionArgs(colorspace = 'YUV',
                                                       orient = 11,
                                                       pix_per_cell = 16,
                                                       cell_per_block = 2,
                                                       hog_channel = 'ALL'))
    feat_ext.extract_features(cars, notcars)
    feat_ext.prepare_features(scale=True)
    
    #test_FeatureClassification_train_classifier(feat_ext, n_predict = min(n_features_max, 10))

    # classifier
    feat_class = FeatureClassification(feat_ext)
    feat_class.train_classifier()

    # Trivial function test
    #test_FeatureClassification_with_trained_classifier(feat_class, n_predict = min(n_features_max, 10))

    # find_cars 1
    test_FeatureClassification_find_cars_single_image(mpimg.imread('./test_images/test1.jpg'), 
                                                      feat_class,
                                                      ystart = 400,
                                                      ystop = 656,
                                                      scale = 1.5)

    
    test_FeatureClassification_find_cars_and_draw_boxes_single_image(mpimg.imread('./test_images/test1.jpg'), 
                                                      feat_class,
                                                      ystart = 400,
                                                      ystop = 656,
                                                      scale = 1.5)

    #for file_name in glob.glob('./test_images/test*.jpg'):
    #    test_FeatureClassification_find_cars_and_draw_boxes_single_image(mpimg.imread(file_name), 
    #                                                      feat_class,
    #                                                      ystart = 400,
    #                                                      ystop = 656,
    #                                                      scale = 1.5)

    test_FeatureClassification_find_cars_multi_scale_single_image(mpimg.imread('./test_images/test1.jpg'), feat_class, xstart=6)

    for file_name in glob.glob('./test_images/test*.jpg'):
        test_FeatureClassification_find_cars_multi_scale_single_image(mpimg.imread(file_name), 
                                                          feat_class,
                                                          xstart=6)
