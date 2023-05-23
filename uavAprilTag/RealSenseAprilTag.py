

#Ryan Upham

#Reference https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
## License: Apache 2.0. See LICENSE file in root directory.


import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag

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

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            #create greyscale image
            grey_image = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2GRAY)

            #detect aprilTags from grey scale image
            results = detector.detect(grey_image)

            for r in results:
                # extract the bounding box (x, y)-coordinates for the AprilTag
                # and convert each of the (x, y)-coordinate pairs to integers
                (ptA, ptB, ptC, ptD) = r.corners
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))
                ptA = (int(ptA[0]), int(ptA[1]))
                # draw the bounding box of the AprilTag detection
                cv2.line(resized_color_image, ptA, ptB, (0, 255, 0), 2)
                cv2.line(resized_color_image, ptB, ptC, (0, 255, 0), 2)
                cv2.line(resized_color_image, ptC, ptD, (0, 255, 0), 2)
                cv2.line(resized_color_image, ptD, ptA, (0, 255, 0), 2)
                # draw the center (x, y)-coordinates of the AprilTag
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(resized_color_image, (cX, cY), 5, (0, 0, 255), -1)
                # draw the tag family on the image
                tagID = str(r.tag_id)
                print(tagID)
                cv2.putText(resized_color_image, tagID, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # find pose of tag
                fndPose, vRot, vTran = cv2.solvePnP(tagSizePnts, r.corners, camMtrx, distortCoefs, flags=0)
                #draw translation vector on image
                dist = np.array2string(vTran)
                cv2.putText(resized_color_image, dist, (ptA[0]+100, ptA[1]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             

            images = np.hstack((resized_color_image, depth_colormap))
        else:
            grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            #detect aprilTags from grey scale image
            results = detector.detect(grey_image)

            for r in results:
                # extract the bounding box (x, y)-coordinates for the AprilTag
                # and convert each of the (x, y)-coordinate pairs to integers
                (ptA, ptB, ptC, ptD) = r.corners
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))
                ptA = (int(ptA[0]), int(ptA[1]))
                # draw the bounding box of the AprilTag detection
                cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
                cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
                cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
                cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)
                # draw the center (x, y)-coordinates of the AprilTag
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
                # draw the tag family on the image
                tagID = str(r.tag_id)
                print(tagID)
                cv2.putText(color_image, tagID, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # find pose of tag
                fndPose, vRot, vTran = cv2.solvePnP(tagSizePnts, r.corners, camMtrx, distortCoefs, flags=0)
                #draw translation vector on image
                dist = np.array2string(vTran)
                cv2.putText(color_image, dist, (ptA[0]+100, ptA[1]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            images = np.hstack((color_image, depth_colormap))


        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()