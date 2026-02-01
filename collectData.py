import cv2 
import os

# start video capture
vid_cap = cv2.VideoCapture(0)

# directory for dataset
directory = 'image/'

while True:
    _, frame = vid_cap.read()
    # dictionary to count number of the images in each subdirectory
    count = {
        'a': len(os.listdir(os.path.join(directory, 'A'))),
        'b': len(os.listdir(os.path.join(directory, 'B'))),
        'c': len(os.listdir(os.path.join(directory, 'C'))),
    }
# get dimensions of each captured image
row= frame.shape[1]
col= frame.shape[0]

# draw a rectangle on the frame
cv2.rectangle(frame, (0 , 40),(300,400),(255,255,255),2)

# display the captured region separately
cv2.imshow('data', frame)
cv2.imshow('ROI', frame[40:400, 0:300])

#crop
frame = frame[40:400, 0:300]

interrupt = cv2.waitKey(10)

if interrupt & 0xFF == ord('a'):
    cv2.imwrite(directory +'A/' + str(count['a']) + '.png', frame)
    count['a'] += 1

if interrupt & 0xFF == ord('b'):
    cv2.imwrite(directory +'B/' + str(count['b']) + '.png', frame)
    count['b'] += 1

if interrupt & 0xFF == ord('c'):
    cv2.imwrite(directory +'C/' + str(count['c']) + '.png', frame)
    count['c'] += 1

# release the video capture device
vid_cap.release()
cv2.destroyAllWindows()