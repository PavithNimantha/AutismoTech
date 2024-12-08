import cv2
import numpy as np
import os
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score 
import joblib


# Load the classifier, class names, scaler, number of clusters and vocabulary 
#from stored pickle file (generated during training)
clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")

print("model loaded");

# Get the path of the testing image(s) and store them in a list
#test_path = 'dataset/test' 
test_path = 'DataSet/archive/images/validation'  # Folder Names are mn and ot
#instead of test if you use train then we get great accuracy

testing_names = os.listdir(test_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
#
def imglist(path):
    return [os.path.join(path, f).replace("\\", "/") for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    print("line 36 : " +testing_name);
    class_id+=1
    
# Create feature extraction and keypoint detector objects
    #SIFT is not available anymore in openCV    
# Create List where all the descriptors will be stored
des_list = []

#BRISK is a good replacement to SIFT. ORB also works but didn;t work well for this example
brisk = cv2.BRISK_create(30)

# Extract descriptors from each image
# Collect only valid image paths
valid_image_paths = []

valid_image_classes = []  # To store the classes of valid images

for image_path, class_id in zip(image_paths, image_classes):
    print("Processing:", image_path)
    im = cv2.imread(image_path)
    if im is None:
        print(f"Warning: Image at {image_path} could not be read. Skipping.")
        continue
    kpts, des = brisk.detectAndCompute(im, None)
    if des is None:
        print(f"Warning: No descriptors found for {image_path}. Skipping.")
        continue
    des_list.append((image_path, des))
    valid_image_paths.append(image_path)  # Add to valid paths
    valid_image_classes.append(class_id)  # Add the corresponding class

# Check if des_list has at least one valid descriptor
if not des_list:
    raise ValueError("No valid descriptors found in the dataset.")

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for _, descriptor in des_list[1:]:
    if descriptor.shape[1] != descriptors.shape[1]:  # Ensure consistency
        print("Warning: Descriptor shape mismatch for an image. Skipping.")
        continue
    descriptors = np.vstack((descriptors, descriptor))

print("Descriptors stacked successfully. Shape:", descriptors.shape)

# kmeans works only on float, so convert integers to float
descriptors_float = descriptors.astype(float)

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

k = 200  # k means with 100 clusters gives lower accuracy for the aeroplane example
voc, variance = kmeans(descriptors_float, k, 1)

# Calculate the histogram of features and represent them as vector
# vq Assigns codes from a code book to observations.
# Ensure im_features matches the length of valid_image_paths
im_features = np.zeros((len(valid_image_paths), k), "float32")

for idx, (image_path, descriptor) in enumerate(des_list):
    if image_path not in valid_image_paths:
        continue
    words, distance = vq(descriptor, voc)
    for w in words:
        im_features[idx][w] += 1
print("line 75 : ");

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# Scale the features
#Standardize features by removing the mean and scaling to unit variance
#Scaler (stdSlr comes from the pickled file we imported)
test_features = stdSlr.transform(im_features)

#######Until here most of the above code is similar to Train except for kmeans clustering####

#Report true class names so they can be compared with predicted classes
true_class = [classes_names[i] for i in valid_image_classes]
# Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf.predict(test_features)]


#Print the true class and Predictions 
print ("true_class ="  + str(true_class))
print ("prediction ="  + str(predictions))

###############################################
#To make it easy to understand the accuracy let us print the confusion matrix

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


accuracy = accuracy_score(true_class, predictions)
print ("accuracy = ", accuracy)
cm = confusion_matrix(valid_image_classes, clf.predict(test_features))
print (cm)

showconfusionmatrix(cm)