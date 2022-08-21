import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy
import os
import cvlib

# Additional libraries:
# pip install opencv-python
# pip install cvlib
# pip install tensorflow
# pip install keras

# Load the model that has been created
genderModel = load_model("genderDetection.model")
genderTypes = [ "Male", "Female" ]
genderColors = [ ( 245, 172, 32 ) , ( 176, 15, 149 ) ]
coordinateMultiplier = [ 0.33, 0.27 ]

# Open the device's webcame to capture vide
captureWebcam = cv2.VideoCapture(0)


def genderDetection(detectedFaces):
    for genderCode, faceCoordinates in enumerate(detectedFaces):
        # The coordinates of the face. 
        x = faceCoordinates[0]
        y = faceCoordinates[1]
        w = faceCoordinates[2]
        h = faceCoordinates[3]
        

        cropFace = numpy.copy( videoFrame[ y:h, x:w ])

        # If a considerable portion of the face is out of the screen, the
        # program will not consider "detecting" it.
        if (cropFace.shape[0]) < 10 or (cropFace.shape[1]) < 10:
            pass
        else:
            # resize the image so that it will conform with the images used
            # to train the model
            cropFace = cv2.resize( cropFace, (96, 96))
            # lessen the number of pixels
            cropFace = cropFace.astype("float") / 255.0
            # convert the pixels into array
            cropFace = img_to_array(cropFace)
            cropFace = numpy.expand_dims( cropFace, axis=0)

            # use the model to predict the gender of the face
            predictionResult = genderModel.predict( cropFace )[0]
            
            # process the results
            genderCode = numpy.argmax(predictionResult)
            gender = genderTypes[genderCode]
            color = genderColors[genderCode]
            

            # Create the rectangles and put the gender text
            cv2.rectangle(videoFrame, (x, y), (w, h), color, 2)
            cv2.rectangle(videoFrame, (x, h+50), (w, h+15), color, -1)
            textCoordinateX = int(((w-x)*coordinateMultiplier[genderCode]) + x) # the x position of the gender text. The multiplier varies depending on the gender
            cv2.putText(videoFrame, gender, (textCoordinateX, h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)      


while True:
    # take the video's frame
    _, videoFrame = captureWebcam.read()

    detectedFaces, detectionConfidence = cvlib.detect_face( videoFrame )

    genderDetection(detectedFaces)
    
    # Output the result
    cv2.imshow( "Face and Gender Recognition", videoFrame )

    # To end the program, press escape
    exitESC = cv2.waitKey(30) & 0xff
    if exitESC == 27:
        break

# Terminate webcam capture
captureWebcam.release()
cv2.destroyAllWindows()

# Additional comments:
# Finished on 3:30pm
# I'll try to deploy this on a website
