import cv2
import numpy as np
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('project_video.mp4')

count = 0
while vidcap.isOpened():
    flag, image = vidcap.read()
    output_image = np.copy(image)
    if flag:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/VOCdevkit/VOC2012/JPEGImages/frame'+str(count)+'.jpg', image)
        count += 1
    else:
        break
    
line_1 = ''
for i in range(count):
    line_1 += 'frame' + str(i) + '\n'

with open('custom.txt', "w") as output:
    output.write(line_1)
    
    
import os

image_folder = 'predictions/VOC2012_custom'
video_name = 'predictions.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()