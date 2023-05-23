#Ryan Upham
#AprilTag Video Tracking
#References:
#   https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
#   https://pyimagesearch.com/2020/11/02/apriltag-with-python/




# import libraries
import cv2
import apriltag
import numpy as np
  
#camera calibration matrix and distortion coefs (zeros)
camMtrx = np.array([[976.07170223,       0,     645.93713226],
 					[ 0,          975.15380297, 338.01976117],
 					[ 0,          0,            1]])
distortCoefs = np.zeros((4,1))

#define tag dimensions in inches, centered on zero
tagSizePnts = np.array([(-3.375,-3.375,0), (-3.375,3.375,0), (3.375,3.375,0), (3.375, -3.375, 0)])

#define april tag detector options 
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    #convert to grey scale  
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect aprilTags from grey scale image
    results = detector.detect(grey)

    # loop over the AprilTag detection results
    for r in results:
	    # extract the bounding box (x, y)-coordinates for the AprilTag
	    # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
	    # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
	    # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
	    # draw the tag family on the image
        tagID = str(r.tag_id)
        print(tagID)
        cv2.putText(frame, tagID, (ptA[0], ptA[1] - 15),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	    # find pose of tag
        fndPose, vRot, vTran = cv2.solvePnP(tagSizePnts, r.corners, camMtrx, distortCoefs, flags=0)
        #draw translation vector on image
        dist = np.array2string(vTran)
        cv2.putText(frame, dist, (ptA[0]+100, ptA[1]-15),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()