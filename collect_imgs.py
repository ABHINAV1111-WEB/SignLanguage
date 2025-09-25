print("Script started")

import os
import cv2

DATA_DIR = 'data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam!")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class', j)
    print('Press "Q" when ready to start collecting images for class', j)

    # Wait for user to press Q to start collecting
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame!")
            break
            
        cv2.putText(frame, f'Ready? Press "Q" for class {j}!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Collect images for this class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame!")
            break
            
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        counter += 1
        print(f"Class {j}: {counter}/{dataset_size}")
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection completed!")