import cv2

print("Testing webcam access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam!")
    exit()

print("Webcam opened successfully!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame!")
        break
    
    cv2.imshow('Webcam Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test completed!") 