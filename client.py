from imutils.video import VideoStream
from mlx90614 import MLX90614
from smbus2 import SMBus
import cv2
import imagezmq

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)

print("[INFO] creating connection...")
sender = imagezmq.ImageSender(connect_to="tcp://10.0.0.162:50007")
print("[INFO] connecting established...")

rpiName = socket.gethostname()
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

jpeg_quality = 80

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    object_temp = sensor.get_object_1()

    msgDict = {
            "rpiName": rpiName,
            "object_temp": object_temp
        }

    ret_code, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    
    reply = sender.send_jpg_reqrep(msgDict, jpg_buffer)

