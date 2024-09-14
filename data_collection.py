import cv2
import os
import time

def create_directory(name):
    directory = name.replace(" ", "_").lower()
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def capture_images(name, num_images=1000):
    directory = create_directory(name)
    cap = cv2.VideoCapture(0)
    
    face_cascade_path = os.path.expanduser('~/haarcascade_frontalface_default.xml')
    
    if not os.path.exists(face_cascade_path):
        print(f"Error: Haar cascade file not found at {face_cascade_path}")
        print("Please make sure you've downloaded the file as instructed.")
        return
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{directory}/image_{count:03d}.jpg", face)
            count += 1
            print(f"Captured image {count}/{num_images}")
        
        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)  #a small delay
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter the name of the person in the pictures: ")
    capture_images(name)
    print(f"Images saved in directory: {name.replace(' ', '_').lower()}")
