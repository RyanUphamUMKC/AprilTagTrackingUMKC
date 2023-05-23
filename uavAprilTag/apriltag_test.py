import cv2
import argparse
import apriltag
import numpy as np


# construct argument for parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "path input image containing apriltag")
args = vars(ap.parse_args())

#load input image and convert to grey scale
print("[INFO] loading image...")	
image = cv2.imread(args["image"])
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#camera calibration stuff
# Params = from camMtrx [(0,0), (1,1), (2,0), (2,1)]
camMtrx = np.array([[976.07170223,       0,     645.93713226],
 					[ 0,          975.15380297, 338.01976117],
 					[ 0,          0,            1]])
tagSizePnts = np.array([(-3.375,-3.375,0), (-3.375,3.375,0), (3.375,3.375,0), (3.375, -3.375, 0)])
distortCoefs = np.zeros((4,1))

#define april tag detector options and detect tags in image
print("[INFO] detecting apriltags")
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
results = detector.detect(grey)
print(results)
print("[INFO] {} total april tags detected".format(len(results)))

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
	cv2.line(image, ptA, ptB, (0, 255, 0), 2)
	cv2.line(image, ptB, ptC, (0, 255, 0), 2)
	cv2.line(image, ptC, ptD, (0, 255, 0), 2)
	cv2.line(image, ptD, ptA, (0, 255, 0), 2)
	# draw the center (x, y)-coordinates of the AprilTag
	(cX, cY) = (int(r.center[0]), int(r.center[1]))
	cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
	# draw the tag family on the image
	tagFamily = r.tag_family.decode("utf-8")
	# find pose of tag
	fndPose, vRot, vTran = cv2.solvePnP(tagSizePnts, r.corners, camMtrx, distortCoefs, flags=0)
	print("[INFO] translation:")
	print(vTran)
	cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	print("[INFO] tag family: {}".format(tagFamily))
	print("[INFO] tagID: {}", r.tag_id)
# show the output image after AprilTag detection
#cv2.imshow("Image", image)
#cv2.waitKey(0)





#
#[DetectionBase(tag_family='tag36h11', tag_id=2, hamming=0, goodness=0.0, decision_margin=98.58241271972656, homography=array([[ -1.41302664e-01,   1.08428082e+00,   1.67512900e+01],
#   [ -8.75899366e-01,   1.50245469e-01,   1.70532040e+00],
#   [ -4.89183533e-04,   2.12210247e-03,   6.02052342e-02]]), center=array([ 278.23643912,   28.32511859]), corners=array([[ 269.8939209 ,   41.50381088],
#   [ 269.57183838,   11.79248142],
#   [ 286.1383667 ,   15.84242821],
#   [ 286.18066406,   43.48323059]])),
#DetectionBase(tag_family='tag36h11', ... etc

