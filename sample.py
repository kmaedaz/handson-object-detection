import numpy as np
from flask import Flask, Response, render_template
from imutils.video.pivideostream import PiVideoStream
import cv2
import time

net = cv2.dnn.readNetFromCaffe('/home/pi/models/MobileNetSSD_deploy.prototxt','/home/pi/models/MobileNetSSD_deploy.caffemodel')

app = Flask(__name__)
camera = PiVideoStream(resolution=(400, 304), framerate=5).start()
time.sleep(2)

def detect(frame):
    frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(
        image=frame,
        scalefactor=0.007843, 
        size=(300, 300),
        mean=127.5
    )
net.setInput(blob)
out = net.forward()   
boxes = out[0,0,:,3:7] * np.array([300, 300, 300, 300])
classes = out[0,0,:,1]
confidences = out[0,0,:,2]

def gen(camera):
    while True:
        frame = camera.read()
        processed_frame = detect(frame.copy())
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        debug=False, 
        threaded=True,
        use_reloader=False
    )
    
