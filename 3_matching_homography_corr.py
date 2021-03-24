import numpy as np
import cv2
import matplotlib.pyplot as plt


left = cv2.VideoCapture('/home/alphat/Desktop/pavin_images/im3/left.mp4')
right = cv2.VideoCapture('/home/alphat/Desktop/pavin_images/im3/right.mp4')

# Create Detector and descriptor, ORB pour changer
orb = cv2.ORB_create(nfeatures=1000)

while(True):
    # Capture frame-by-frame
    lret, lframe = left.read()
    rret, rframe = right.read()
    if lret == False or rret == False:
        break
    
    # Convert to grayscale
    # lgray = cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY)
    # rgray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    
    #---------------------------------
    # Create Keypoints and descriptors
    #---------------------------------    
    # Compute interest points and descriptors
    kp_left, ds_left = orb.detectAndCompute(lframe, None)
    kp_right, ds_right = orb.detectAndCompute(rframe, None)
    
    #---------------------------------
    # Matching
    #---------------------------------
    # BFMatcher with default params
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ds_left, ds_right)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    factor = 3
    n_matchs = 0
    
    if matches:
        threshold = factor * matches[0].distance
        for m in matches:
            if m.distance < threshold:
                n_matchs = n_matchs+1
            else:
                break
        print('nb matches', n_matchs)    
    
    #---------------------------------
    # Compute homography
    #---------------------------------
    if n_matchs > 4: # at least 4 matches for homography estimation 
        src_pts = np.float32([kp_left[m.queryIdx].pt for m in matches[0:n_matchs]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_right[m.trainIdx].pt for m in matches[0:n_matchs]]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        print('H = ', H)
      
    #---------------------------------
    # Compute rectangle
    #---------------------------------
        # src_rect = np.array([[0, 0],
        #                     [imgref.shape[1],0],
        #                     [imgref.shape[1],imgref.shape[0]],
        #                     [0,imgref.shape[0]]])
        # cur_rect = cv2.perspectiveTransform(src_rect.reshape(-1,1,2).astype(np.float32), H)

    #---------------------------------
    # Display 
    #---------------------------------
    # Rectangle : 
    # frame = cv2.polylines(lframe, [np.int32(cur_rect)], True, 255, 3, cv2.LINE_AA)
    
    # Matching results
    imgmatching = cv2.drawMatches(lframe, kp_left, rframe, kp_right, matches[0:n_matchs], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the resulting frame
    cv2.imshow('ref image', lframe)
    cv2.imshow('color frame', rframe)
    cv2.imshow('Matching result', imgmatching)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
left.release()
right.release()
cv2.destroyAllWindows()
