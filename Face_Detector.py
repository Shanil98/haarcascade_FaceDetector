''' WE USE HAAR CASCADE METHOD TO MAKE A FACE DETECTOR '''

# open source computer vision library
import cv2
from random import randrange

# lets load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('trained_face_recognizer.xml')

# choose an image to detect faces in
#img = cv2.imread('adele3.jpeg')

# To capture video from webcam, the 0 as the arg is the default computer camera, otherwise we can put
# a video file name in there 
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()
    # ^^ is just boolean,  ^^ is the actual video frame

    # lets convert the image into grayscale format
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # let us now detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangles around the faces
    for (x,y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x ,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video playing now', frame)
    #getting the key pressed at that millisecond
    key = cv2.waitKey(1)
    
    # checking is the keyboard key pressed is Q for quitting the window
    # checking key agains ASCII for Q or q
    if key == 81 or key == 113:
        break

# Release the VideoCapture object, this kind of cleans up the code, kinda like closing the file after
# opening it
webcam.release() 
# print(face_coordinates)

print('program ended')
