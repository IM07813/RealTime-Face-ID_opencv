import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import dlib
import pickle
from collections import Counter

def load_images(directory):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person)
        if not os.path.isdir(person_dir):
            continue
        
        if person not in label_dict:
            label_dict[person] = current_label
            current_label += 1
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(label_dict[person])

    return images, labels, label_dict

def extract_features(images, labels):
    face_detector = dlib.get_frontal_face_detector()
    
    shape_predictor_path = os.path.expanduser('~/shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(shape_predictor_path):
        print(f"Error: Shape predictor file not found at {shape_predictor_path}")
        print("Please make sure your python file training.py is in the same directory.")
        return None, None

    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    
    face_rec_model_path = os.path.expanduser('~/dlib_face_recognition_resnet_model_v1.dat')
    if not os.path.exists(face_rec_model_path):
        print(f"Error: Face recognition model file not found at {face_rec_model_path}")
        print("Please make sure your python file training.py is in the same directory.")
        return None, None

    face_recognition_model = dlib.face_recognition_model_v1(face_rec_model_path)

    features = []
    valid_labels = []
    for image, label in zip(images, labels):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        
        if len(faces) > 0:
            shape = shape_predictor(gray, faces[0])
            face_descriptor = face_recognition_model.compute_face_descriptor(image, shape)
            features.append(face_descriptor)
            valid_labels.append(label)
        else:
            print(f"Warning: No face detected in an image. Skipping this image.")

    return features, valid_labels

def train_model(features, labels):
    if len(set(labels)) < 2:
        print("Error: At least two different people are required for training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return clf

if __name__ == "__main__":
    data_directory = "."  # person directory should be a subdirectory of this directory where training.py resides 
    images, labels, label_dict = load_images(data_directory)
    
    if not images:
        print("No images found. Please check the data directory.")
        exit()

    print("Extracting features...")
    features, valid_labels = extract_features(images, labels)
    
    if features is None or valid_labels is None:
        print("Feature extraction failed.")
        exit()

    if len(features) == 0:
        print("No faces detected in any of the images. Please check your dataset.")
        exit()

    print(f"Successfully extracted features from {len(features)} images.")
    
    # Count the number of images for each person
    label_counts = Counter(valid_labels)
    print("\nDataset composition:")
    for label, count in label_counts.items():
        person_name = [name for name, idx in label_dict.items() if idx == label][0]
        print(f"  {person_name}: {count} images")

    if len(label_counts) < 2:
        print("\nError: At least two different people are required for training.")
        print("Please add images of at least one more person to your dataset.")
        exit()

    print("\nTraining model...")
    model = train_model(features, valid_labels)
    
    if model is None:
        print("Model training failed. Please check the error messages above.")
        exit()

    # Save the model and label dictionary
    with open("face_recognition_model.pkl", "wb") as f:
        pickle.dump((model, label_dict), f)
    
    print("Model and label dictionary saved as 'face_recognition_model.pkl'")
    
    # Print the label dictionary for verification
    print("\nLabel Dictionary:")
    for person, label in label_dict.items():
        print(f"  {person}: {label}")
