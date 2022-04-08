from skimage.registration import phase_cross_correlation
import numpy as np
import cv2

def RegisterDFT(images,base_frame):
	resFactor=8;
	images=np.array(images)
	N,w,h = images.shape
	baseGray=np.array(base_frame)
	
	
	shifted_images=[]
	for i in range(N):
		nextGray= images[i,:,:]
		shift, error, diffphase = phase_cross_correlation(baseGray, nextGray,upsample_factor=resFactor)
		
		M = np.float32([[1, 0, shift[1]],[0, 1, shift[0]]])
		
		shifted = cv2.warpAffine(nextGray, M, (h,w))

		shifted_images.append(shifted.astype(np.float32))
		
	
	return np.array([shifted_images])