import cv2
import dlib
import numpy as np
import pickle
import os
import time
from collections import Counter

def load_model():
    with open("face_recognition_model.pkl", "rb") as f:
        model, label_dict = pickle.load(f)
    return model, {v: k for k, v in label_dict.items()}

def draw_face_mesh(image, shape):
    for i in range(68):
        x, y = shape.part(i).x, shape.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    mesh_lines = [
        (0, 16), (16, 26), (26, 17), (17, 0),  
        (0, 36), (36, 39), (39, 42), (42, 45), (45, 16),  
        (17, 21), (21, 22), (22, 26),  
        (22, 23), (23, 24), (24, 25), (25, 26),  
        (27, 30), (30, 33), (33, 35), (35, 30),  
        (48, 54), (54, 60), (60, 64), (64, 48)  
    ]
    
    for start, end in mesh_lines:
        start_point = (shape.part(start).x, shape.part(start).y)
        end_point = (shape.part(end).x, shape.part(end).y)
        cv2.line(image, start_point, end_point, (0, 255, 0), 1)

def estimate_distance(face):
    "note: its a rough approximation, still working on it" 
    face_width = face.right() - face.left()
    distance = 1000 / face_width  # in arbitrary units
    return distance

def get_light_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def main():
    model, label_dict = load_model()
    
    face_detector = dlib.get_frontal_face_detector()
    
    shape_predictor_path = os.path.expanduser('~/shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(shape_predictor_path):
        print(f"Error: Shape predictor file not found at {shape_predictor_path}")
        return
    
    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    
    face_rec_model_path = os.path.expanduser('~/dlib_face_recognition_resnet_model_v1.dat')
    if not os.path.exists(face_rec_model_path):
        print(f"Error: Face recognition model file not found at {face_rec_model_path}")
        return
    
    face_recognition_model = dlib.face_recognition_model_v1(face_rec_model_path)
    
    cap = cv2.VideoCapture(0)
    
    # Set a larger window size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    PROBABILITY_THRESHOLD = 0.8
    EVALUATION_TIME = 10  # seconds
    DISPLAY_TIME = 3  # seconds
    
    while True:
        start_time = time.time()
        predictions = []
        
        # Capture and process frames for EVALUATION_TIME seconds
        while time.time() - start_time < EVALUATION_TIME:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            
            light_intensity = get_light_intensity(frame)
            cv2.putText(frame, f"Light Intensity: {light_intensity:.2f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for face in faces:
                shape = shape_predictor(gray, face)
                face_descriptor = face_recognition_model.compute_face_descriptor(frame, shape)
                
                prediction = model.predict_proba([face_descriptor])[0]
                best_class_index = np.argmax(prediction)
                best_class_probability = prediction[best_class_index]
                
                if best_class_probability < PROBABILITY_THRESHOLD:
                    predictions.append("Unknown Person")
                else:
                    predictions.append(label_dict[best_class_index])
                
                # Draw bounding box and face mesh
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                draw_face_mesh(frame, shape)
                
                # Estimate and display distance
                distance = estimate_distance(face)
                cv2.putText(frame, f"Distance: {distance:.2f}", (left, bottom + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Determine the most common prediction
        if predictions:
            most_common_prediction = Counter(predictions).most_common(1)[0][0]
            confidence = Counter(predictions)[most_common_prediction] / len(predictions)
            
            if confidence >= 0.75:
                print(f"Predicted: {most_common_prediction} (Confidence: {confidence:.2f})")
                
                # Display the result for DISPLAY_TIME seconds
                display_start = time.time()
                while time.time() - display_start < DISPLAY_TIME:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    light_intensity = get_light_intensity(frame)
                    cv2.putText(frame, f"Light Intensity: {light_intensity:.2f}", (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.putText(frame, f"{most_common_prediction}: {confidence:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Face Recognition", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            else:
                print("No confident prediction made.")
        else:
            print("No faces detected during the evaluation period.")

if __name__ == "__main__":
    main()
