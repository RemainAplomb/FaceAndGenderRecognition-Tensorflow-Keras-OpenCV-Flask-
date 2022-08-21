import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy
import os
import cvlib

class webcamVideo():
    def __init__(self):
        # Open the device's webcame to capture video
        self.captureWebcam = cv2.VideoCapture(0)

        # Load the model that has been created
        self.genderModel = load_model("genderDetection.model")
        self.genderTypes = [ "Male", "Female" ]
        self.genderColors = [ ( 245, 172, 32 ) , ( 176, 15, 149 ) ]
        self.coordinateMultiplier = [ 0.33, 0.27 ]

    def __del__(self):
        # Terminate webcam capture
        self.captureWebcam.release()

    def genderDetection(self):
        for genderCode, faceCoordinates in enumerate(self.detectedFaces):
            x = faceCoordinates[0]
            y = faceCoordinates[1]
            w = faceCoordinates[2]
            h = faceCoordinates[3]

            self.cropFace = numpy.copy( self.videoFrame[ y:h, x:w ])

            # If a considerable portion of the face is out of the screen, the
            # program will not consider "detecting" it.
            if (self.cropFace.shape[0]) < 10 or (self.cropFace.shape[1]) < 10:
                pass
            else:
                # resize the image so that it will conform with the images used
                # to train the model
                self.cropFace = cv2.resize( self.cropFace, (96, 96))
                # lessen the number of pixels
                self.cropFace = self.cropFace.astype("float") / 255.0
                # convert the pixels into array
                self.cropFace = img_to_array(self.cropFace)
                self.cropFace = numpy.expand_dims( self.cropFace, axis=0)

                # use the model to predict the gender of the face
                self.predictionResult = self.genderModel.predict( self.cropFace )[0]
                
                # process the results
                self.genderCode = numpy.argmax(self.predictionResult)
                self.gender = self.genderTypes[self.genderCode]
                self.color = self.genderColors[self.genderCode]
                

                # Create the rectangles and put the gender text
                cv2.rectangle(self.videoFrame, (x, y), (w, h), self.color, 2)
                cv2.rectangle(self.videoFrame, (x, h+50), (w, h+15), self.color, -1)
                self.textCoordinateX = int(((w-x)*self.coordinateMultiplier[self.genderCode]) + x) # the x position of the gender text. The multiplier varies depending on the gender
                cv2.putText(self.videoFrame, self.gender, (self.textCoordinateX, h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        return

    def getFrame(self):

        # take the video's frame
        _, self.videoFrame = self.captureWebcam.read()
        self.detectedFaces, self.detectionConfidence = cvlib.detect_face( self.videoFrame )

        self.genderDetection()

        """
        # Output the result
        cv2.imshow( "img", self.videoFrame )
        """

        _, self.outputFrame = cv2.imencode( ".jpg", self.videoFrame )
        return self.outputFrame.tobytes()
