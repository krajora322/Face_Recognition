import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'images'
images = []
personNames = []
myList = os.listdir(path)
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(0) == 13:
        break

cap.release()
cv2.destroyAllWindows()


##2nd code


import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

# Define the directory for training images
path = os.path.join(os.getcwd(), "Training_images")

# Ensure the directory exists
if not os.path.exists(path):
    print(f"Error: The directory '{path}' does not exist. Creating it now...")
    os.makedirs(path)

# Load images and store encodings
def load_training_images(directory):
    images = []
    classNames = []
    imageList = os.listdir(directory)
    
    for imgName in imageList:
        imgPath = os.path.join(directory, imgName)
        img = cv2.imread(imgPath)
        
        if img is None:
            print(f"Warning: '{imgName}' is not a valid image file. Skipping.")
            continue
        
        images.append(img)
        classNames.append(os.path.splitext(imgName)[0])  # Remove file extension
    
    return images, classNames

# Get encodings for all images
def encode_faces(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)
        if encoding:
            encodedList.append(encoding[0])
    return encodedList

# Mark attendance in a CSV file
def mark_attendance(name):
    file_path = 'Attendance.csv'
    with open(file_path, 'a+') as f:
        f.seek(0)
        existing_entries = f.readlines()
        
        # Check if the name already exists in today's attendance
        now = datetime.now()
        today_date = now.strftime('%Y-%m-%d')
        already_present = any(name in entry and today_date in entry for entry in existing_entries)

        if not already_present:
            time_now = now.strftime('%H:%M:%S')
            f.write(f'{name},{today_date},{time_now}\n')
            print(f"Attendance marked for: {name}")

# Main function for face recognition
def recognize_faces():
    images, classNames = load_training_images(path)
    knownEncodings = encode_faces(images)

    print("Encoding complete. Starting webcam...")
    
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Unable to access the webcam.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces_in_frame = face_recognition.face_locations(small_frame)
        encodes_in_frame = face_recognition.face_encodings(small_frame, faces_in_frame)

        for encodeFace, faceLoc in zip(encodes_in_frame, faces_in_frame):
            matches = face_recognition.compare_faces(knownEncodings, encodeFace)
            faceDistances = face_recognition.face_distance(knownEncodings, encodeFace)

            best_match_index = np.argmin(faceDistances) if faceDistances else None

            if best_match_index is not None and matches[best_match_index]:
                name = classNames[best_match_index].upper()
                mark_attendance(name)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the face recognition system
if _name_ == "_main_":
    recognize_faces()
