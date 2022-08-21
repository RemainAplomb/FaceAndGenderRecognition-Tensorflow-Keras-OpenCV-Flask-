from flask import Flask, render_template, Response
from mainFace import webcamVideo

app = Flask(__name__)

@app.route('/')

def index():
    return render_template("index.html")

def generateFrame(mainFace):
    while True:
        frame=mainFace.getFrame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')
        
@app.route('/video')
def websiteVideo():
    return Response(generateFrame(webcamVideo()),
    mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(debug=True)
