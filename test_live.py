import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
print("Loading trained model...")
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

print("Model loaded successfully!")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define class labels
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

print("Starting real-time sign language recognition...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame!")
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract landmarks for prediction
            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)
            
            # Make prediction
            prediction = model.predict([data_aux])
            predicted_class = int(prediction[0])
            
            # Get prediction probability
            prediction_proba = model.predict_proba([data_aux])
            confidence = np.max(prediction_proba) * 100
            
            # Display prediction
            label = labels_dict[predicted_class]
            cv2.putText(frame, f'{label} ({confidence:.1f}%)', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Display instructions
    cv2.putText(frame, 'Show hand sign to classify', (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)
    
    # Break on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Recognition stopped!") 