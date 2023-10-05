import cv2
import dlib

# Load the pre-trained facial landmark predictor
predictor_path = (r'C:\Users\darsa\Downloads\shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)

#let us load the pretrained face detecctor
detector = dlib.get_frontal_face_detector()

# define the function to calculate the eye-gaze direction
def calculate_eye_gaze(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    left_eye_center = (sum([pt.x for pt in left_eye])//6, sum([pt.y for pt in left_eye])//6)
    right_eye_center = (sum([pt.x for pt in right_eye])//6, sum([pt.y for pt in right_eye])//6)

    eye_gaze_vector = (right_eye_center[0]-left_eye_center[0], right_eye_center[1]-left_eye_center[0])

    return eye_gaze_vector

# defining the function that checks if you are distracted or not
def check_distracted(eye_gaze_vector):
    # if the eye_gaze vector[0] is positive, then the person might be looking towards the right
    if eye_gaze_vector[0] >20:
        return 'Distracted'
    else:
        return 'Not distracted'

# Now we need to initialize the video
video_capture = cv2.VideoCapture(0)

while True:
    # we need to capture frame-by-frame
    ret, frame = video_capture.read()
    # we need to convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # we need to detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # we need to detect the facial landmarks for the detected face
        landmarks = predictor(gray,face)
        # calculate eye_gaze direction
        eye_gaze_vector = calculate_eye_gaze(landmarks.parts())
        # the next step is to check if the person is distracted or not
        status_distracted = check_distracted(eye_gaze_vector)
        # distracted display status on frame
        cv2.putText(frame, status_distracted, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (213,125,127), 2)
        # display resulting frame
        cv2.imshow('Eye-gaze tracking for attention', frame)
        # exit the loop if 'e' is pressed
        if cv2.waitKey(1) & 0xFF ==ord('e'):
            break

video_capture.release()
cv2.destroyAllWindows()






