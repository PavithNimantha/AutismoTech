import cv2
import numpy as np
import os

# Get the training classes names and store them in a list
# Here we use folder names for class names


# add the training folder path here
train_path = 'DataSet/archive/images/train'  # Folder Names will be "Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised"
training_names = os.listdir(train_path)



# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0


# To make it easy to list all file names in a directory
#
def imglist(path):
    return [os.path.join(path, f).replace("\\", "/") for f in os.listdir(path)]


# Fill the placeholder empty lists with image path, classes, and add class ID number
#

for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    print("line 36 : " +training_name);
    class_id += 1

# Create feature extraction and keypoint detector objects
# Create List where all the descriptors will be stored
des_list = []

brisk = cv2.BRISK_create(30)


# Collect only valid image paths
valid_image_paths = []

# To store the classes of valid images
valid_image_classes = []  

# Extract descriptors from each image
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

# kmeans works only on float convert integers to float
descriptors_float = descriptors.astype(float)

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

k = 1000  # k means with 100 clusters gives lower accuracy for the aeroplane example
voc, variance = kmeans(descriptors_float, k, 1)

# Calculate the histogram of features and represent them as vector
im_features = np.zeros((len(valid_image_paths), k), "float32")

for idx, (image_path, descriptor) in enumerate(des_list):
    words, distance = vq(descriptor, voc)
    for w in words:
        im_features[idx][w] += 1
        
# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# Scaling the words
# Standardize features by removing the mean and scaling to unit variance
# In a way normalization
from sklearn.preprocessing import StandardScaler

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train an algorithm to discriminate vectors corresponding to positive and negative training images
# Train the Linear SVM
from sklearn.svm import LinearSVC

clf = LinearSVC(max_iter=100000)  # Default of 100 is not converging
clf.fit(im_features, np.array(valid_image_classes))

# Train Random forest to compare how it does against SVM
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators = 100, random_state=30)
# clf.fit(im_features, np.array(image_classes))


# Save the SVM
import joblib

joblib.dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)
