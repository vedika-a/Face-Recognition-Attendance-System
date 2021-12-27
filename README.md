# Face-Recognition-Attendance-System
I developed this project using python language.A simple python script that recognises faces and mark attendance for the recognised faces in an excel sheet.
Face-Recognition-Attendance-System
A simple python script that recognises faces and mark attendance for the recognised faces in an excel sheet. You need following libraries pre-installed on your system: 
1.face_recognition 
2.Opencv 
3.openpyxl 
4.datetime

Steps to develop face recognition model Before moving on, let’s know what face recognition and detection are. Face recognition is the process of identifying or verifying a person’s face from photos and video frames. Face detection is defined as the process of locating and extracting faces (location and size) in an image for use by a face detection algorithm. Face recognition method is used to locate features in the image that are uniquely specified. The facial picture has already been removed, cropped, scaled, and converted to grayscale in most cases. Face recognition involves 3 steps: face detection, feature extraction, face recognition. OpenCV is an open-source library written in C++. It contains the implementation of various algorithms and deep neural networks used for computer vision tasks.

Prepare the dataset Create 2 directories, train and test. Pick an image for each of the cast from the internet and download it onto our “train” directory. Make sure that the images you’ve selected show the features of the face well enough for the classifier.
For testing the model, let’s take a picture containing all of the cast and place it onto our “test” directory.

For your comfort, we have added training and testing data with the project code.

Train the model First import the necessary modules. import face_recognition as fr import cv2 import numpy as np import os The face_recognition library contains the implementation of the various utilities that help in the process of face recognition. Now, create 2 lists that store the names of the images (persons) and their respective face encodings. path = "./train/" known_names = [] known_name_encodings = [] images = os.listdir(path) Face encoding is a vector of values representing the important measurements between distinguishing features of a face like the distance between the eyes, the width of the forehead, etc. We loop through each of the images in our train directory, extract the name of the person in the image, calculate its face encoding vector and store the information in the respective lists. for _ in images: image = fr.load_image_file(path + _) image_path = path + _ encoding = fr.face_encodings(image)[0] known_name_encodings.append(encoding) known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())
Test the model on the test dataset As mentioned above, our test dataset only contains 1 image with all of the persons in it.
Read the test image using the cv2 imread() method.

test_image = "./test/test.jpg" image = cv2.imread(test_image) The face_recognition library provides a useful method called face_locations() which locates the coordinates (left, bottom, right, top) of every face detected in the image. Using those location values we can easily find the face encodings.

face_locations = fr.face_locations(image) face_encodings = fr.face_encodings(image, face_locations) We loop through each of the face locations and its encoding found in the image. Then we compare this encoding with the encodings of the faces from the “train” dataset.

Then calculate the facial distance meaning that we calculate the similarity between the encoding of the test image and that of the train images. Now, we pick the minimum valued distance from it indicating that this face of the test image is one of the persons from the training dataset.

Now, draw a rectangle with the face location coordinates using the methods from the cv2 module.

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings): matches = fr.compare_faces(known_name_encodings, face_encoding) name = "" face_distances = fr.face_distance(known_name_encodings, face_encoding) best_match = np.argmin(face_distances) if matches[best_match]: name = known_names[best_match] cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2) cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED) font = cv2.FONT_HERSHEY_DUPLEX cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) Display the image using the imshow() method of the cv2 module. cv2.imshow("Result", image) Save the image to our current working directory using the imwrite() method.

cv2.imwrite("./output.jpg", image) Release the resources that weren’t deallocated(if any).

cv2.waitKey(0) cv2.destroyAllWindows() Python Face Recognition Output
