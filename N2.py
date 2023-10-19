import cv2



# Load pre-trained Haar Cascade classifiers for face, eyes, nose, and mouth
fcascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(fcascPath)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

ncascPath = "haarcascade_mcs_nose.xml"
noseCascade = cv2.CascadeClassifier(ncascPath)

mcascPath = "haarcascade_mcs_mouth.xml"
mouthCascade = cv2.CascadeClassifier(mcascPath)

# Load the input image or video
input_path = 0  # Replace with your image or video path
cap = cv2.VideoCapture(input_path)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    
    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) within the face
        roi_gray = gray[y:y + h, x:x + w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        
        # Detect nose within the face ROI
        noses = noseCascade.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(frame, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (0, 0, 255), 2)
        
        # Detect mouth within the face ROI
        mouths = mouthCascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(frame, (x + mx, y + my), (x + mx + mw, y + my + mh), (255, 255, 0), 2)
    
    # Display the frame with detected features
    cv2.imshow('Face and Facial Features Detection', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
