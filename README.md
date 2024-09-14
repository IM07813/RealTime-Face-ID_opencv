# Face Recognition System

This project is a **real-time face recognition system** that captures, trains, and recognizes faces using OpenCV, Dlib, and Support Vector Machine (SVM) for classification. The system captures images, trains a model using a facial feature extraction method, and recognizes faces in real time using a web camera.

## Features

- **Image Capture**: Captures images from a webcam and stores them in a specified folder for each person.
- **Feature Extraction**: Extracts facial features using Dlib's face recognition model and 68-face landmarks.
- **Training**: Trains an SVM classifier to identify the faces using extracted features.
- **Face Recognition**: Recognizes faces in real time by comparing webcam footage with the trained model.
- **Additional Features**: 
  - Detects lighting intensity.
  - Estimates rough distance of the person from the camera.
  - Displays a face mesh over detected faces.

## Installation

Follow the steps below to get the project up and running on your local machine.

### Prerequisites

- Python 3.7+
- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)
- Scikit-learn
- NumPy
- Pickle

### Install Required Libraries

```bash
pip install opencv-python dlib scikit-learn numpy
```

Make sure you have the necessary pre-trained models from Dlib:

- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`
- `haarcascade_frontalface_default.xml` (from OpenCV for face detection)

You can download them from:

- [Dlib models](http://dlib.net/).
- [Haar Cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).

### Directory Structure

Ensure your project folder looks something like this:

```bash
/project
  |-- data_collection.py
  |-- training.py
  |-- recognition.py
  |-- shape_predictor_68_face_landmarks.dat
  |-- dlib_face_recognition_resnet_model_v1.dat
  |-- haarcascade_frontalface_default.xml
  |-- face_recognition_model.pkl  # generated after training
  |-- captured_images/            # created by data_collection.py
```

### Step 1: Capture Images

Before training the model, you need to capture images for each person to be recognized.

```bash
python data_collection.py
```

- Enter the name of the person when prompted. This will create a folder under `captured_images/` with that name, where the images will be stored.
- The script will start capturing images from your webcam and save them into the folder.
- It uses the Haar Cascade for face detection, draws a rectangle around detected faces, and saves the face as an image.

### Step 2: Train the Model

Once images are captured, use the `training.py` script to train the face recognition model.

```bash
python training.py
```

- The script loads all images from the `captured_images/` folder, extracts facial features using Dlib’s model, and trains an SVM classifier.
- After training, the model is saved as `face_recognition_model.pkl`, and a label dictionary is generated for mapping person names to labels.

### Step 3: Real-time Face Recognition

With the trained model, you can now recognize faces in real time using the webcam.

```bash
python recognition.py
```

- The script loads the trained model and label dictionary.
- The webcam captures frames in real-time and recognizes any faces detected. The recognized face name, along with a confidence score, is displayed on the screen.
- You can press the `q` key to quit the program.

## Code Details

### 1. Data Collection

- **File**: `data_collection.py`
- **Purpose**: Captures images of a person using the webcam.
- **Key Functions**:
  - `create_directory(name)`: Creates a folder to store images.
  - `capture_images(name, num_images=1000)`: Captures face images and saves them in the created directory.

### 2. Training

- **File**: `training.py`
- **Purpose**: Trains a face recognition model using images captured in the previous step.
- **Key Functions**:
  - `load_images(directory)`: Loads images and their labels from the given directory.
  - `extract_features(images, labels)`: Extracts face descriptors from the images using Dlib.
  - `train_model(features, labels)`: Trains an SVM classifier on the extracted face descriptors.

### 3. Face Recognition

- **File**: `recognition.py`
- **Purpose**: Performs real-time face recognition using the trained model.
- **Key Functions**:
  - `load_model()`: Loads the pre-trained model and label dictionary.
  - `estimate_distance(face)`: Roughly estimates the distance of the face from the camera.
  - `get_light_intensity(image)`: Calculates the light intensity of the image frame.
  - `draw_face_mesh(image, shape)`: Draws a face mesh with 68 landmarks on the detected face.

## Future Improvements

- Improve distance estimation based on actual physical measurements.
- Add more robust handling for different lighting conditions.
- Incorporate more advanced classification techniques such as deep learning models (e.g., CNNs).
- Enhance the user interface for better visualization of predictions and confidence scores.

## Troubleshooting

- **Haar Cascade/Dlib Models Not Found**: Ensure you’ve downloaded the required models and placed them in the correct directory.
- **No Faces Detected**: Make sure your webcam is working correctly and the lighting is sufficient for face detection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
