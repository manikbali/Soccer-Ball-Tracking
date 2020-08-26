import cv2
import numpy as np
import shutil,os

video="8.mp4"

#cap.release()
#os.remove(r"C:\Users\Navya\Documents\28.mp4")
shutil.copy(r"C:/Users/Navya/Documents/ball tracking/ball tracking/8.mp4", r"C:/Users/Navya/Documents" )
#shutil.copy(r"C:/Users/Navya/Documents/ball tracking/ball tracking/12.mp4", r"C:/Users/Navya/Documents" )
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(r"c:\Users\Navya\Documents\multi-object-tracking\multi-object-tracking\videos\soccer_02.mp4")
#cap = cv2.VideoCapture(r"C:\Users\Navya\Documents\12.mp4")
cnt=0

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

ret,first_frame = cap.read()

# Read until video is completed
while(cap.isOpened()):
    
  # Capture frame-by-frame
  ret, frame = cap.read()
     
  if ret == True:
    
    #removing scorecard
    roi = frame[430:-1,140:-200]
#    cv2.imshow("image",roi)
#    exit()
    #cropping center of an image
#    thresh=60
    thresh=0
    end = roi.shape[1] - thresh
    roi = roi[:,thresh:end]
    
    cv2.imshow("image",roi)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    cv2.imwrite('fframes/'+str(cnt)+'.png',roi)
    cnt=cnt+1

  # Break the loop
  else: 
    break
cap.release()
#os.remove(r"C:\Users\Navya\Documents\12.mp4")
cv2.destroyAllWindows()    