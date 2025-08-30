import cv2
import matplotlib.pyplot as plt




config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


model = cv2.dnn_DetectionModel(frozen_model,config_file)


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


classLabels = []
file_name = 'Labels.txt'
with open(file_name,'rt') as fpt :
    classLabels = fpt.read().rstrip('\n').split('\n')




print(classLabels)


print(len(classLabels))


# img = cv2.imread('manbmw.jpg')

# cv2.imshow("",img)

# # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# ClassIndex, confidence, bbox = model.detect(img,confThreshold = 0.5)


# print(ClassIndex)

# font_scale = 1.5
# font = cv2.FONT_HERSHEY_PLAIN
# for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
#     cv2.rectangle(img,boxes,(255,0,0),2)
#     cv2.putText(img,classLabels[ClassInd-1],(boxes[0]-10,boxes[1]+10),font,fontScale=font_scale,color=(0,255,0),thickness=2)


# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# cv2.imshow("",img)

cap = cv2.VideoCapture('vid1.mp4')
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open video source")

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:   # camera not giving frames
        print("No frame captured, exiting...")
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= len(classLabels):
                x, y, w, h = boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (x, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow('OBJECT DETECTION', frame)

    # Exit on 'q'
    if cv2.waitKey(2) & 0xFF == ord('q'):
        print("Exit requested by user")
        break

cap.release()
cv2.destroyAllWindows()       

