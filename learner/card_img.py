"""
Card Recognition using OpenCV
Code from the blog post 
http://arnab.org/blog/so-i-suck-24-automating-card-games-using-opencv-and-python

Usage: 

  ./card_img.py filename num_cards training_image_filename training_labels_filename num_training_cards

Example:
  ./card_img.py test.JPG 4 train.png train.tsv 56
  
Note: The recognition method is not very robust; please see SIFT / SURF for a good algorithm.  

"""

import sys
import numpy as np
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/") 
import cv2


###############################################################################
# Utility code from 
# http://git.io/vGi60A
# Thanks to author of the sudoku example for the wonderful blog posts!
###############################################################################

def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew

###############################################################################
# Image Matching
###############################################################################
def preprocess(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2 )
  thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
  return thresh
  
def imgdiff(img1,img2):
  img1 = cv2.GaussianBlur(img1,(5,5),5)
  img2 = cv2.GaussianBlur(img2,(5,5),5)    
  diff = cv2.absdiff(img1,img2)  
  diff = cv2.GaussianBlur(diff,(5,5),5)    
  flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
#****ONE CARD
  '''
  cv2.namedWindow('Gray')
  img2 = cv2.resize(img2,(426,426))
  cv2.moveWindow('Gray',0,437)
  cv2.imshow('Gray',img2) #todo *********Tear out window, wrong place, probably wrong function. try get cards.
  '''
#END PRINT ONE CARD
  return np.sum(diff)  

def find_closest_card(training,img):
  features = preprocess(img)
#****not this one
  """
  cv2.namedWindow('Gray')
  img = cv2.resize(img,(426,426))
  cv2.moveWindow('Gray',0,437)
  cv2.imshow('Gray',img) #todo *********Tear out window, wrong place, probably wrong function. try get cards.
  """
#END PRINT ONE CARD
  return sorted(training.values(), key=lambda x:imgdiff(x[1],features))[0][0]
  
   
###############################################################################
# Card Extraction
###############################################################################  
def getCards(im, numcards=4): ############################################################  numcards=4????
  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 
       
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]  

  for card in contours:
    peri = cv2.arcLength(card,True)
    approx = rectify(cv2.approxPolyDP(card,.02*peri,True))
	
#There were comments 82-85
    box = np.int0(approx)
    cv2.drawContours(im,[box],0,(255,255,0),6)
    imx = cv2.resize(im,(854,854)) #1000,600
    cv2.namedWindow('test2')
    cv2.moveWindow('test2',426,0)
    cv2.imshow('test2',imx)
#  cv2.imshow('a',imx)      
    
    h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

    transform = cv2.getPerspectiveTransform(approx,h)
    warp = cv2.warpPerspective(im,transform,(450,450))
    
    yield warp





def get_training():
	training = {}
	labels = {}

	for line in file('train_new.tsv'):
		key, num, suit = line.strip().split()
		labels[int(key)] = (num,suit)

	print "Training"

	for i in range(1,53):
		if (i==37):
			continue
		filename = 'JcardPics/%s.jpg'%i
		print filename
		im=cv2.imread(filename)
		for j,c in enumerate(getCards(im,1)):
			pass
		training[i] = (labels[i], preprocess(c))
	print "Done Training"
	return training
	
	

"""def get_training(training_labels_filename,training_image_filename,num_training_cards,avoid_cards=None):
  training = {}
  
  labels = {}
  for line in file(training_labels_filename): 
    key, num, suit = line.strip().split()
    labels[int(key)] = (num,suit)
    
  print "Training"

  im = cv2.imread(training_image_filename)
  for i,c in enumerate(getCards(im,num_training_cards)):
    if avoid_cards is None or (labels[i][0] not in avoid_cards[0] and labels[i][1] not in avoid_cards[1]):
      training[i] = (labels[i], preprocess(c))
  
  print "Done training"
  return training"""
  

if __name__ == '__main__':
  if len(sys.argv) == 2:		#CHANGE ME
    filename = sys.argv[1]		#CHANGE ME  to the JPEG used, if CV img is used see line 146

    training = get_training()
    print filename
    im = cv2.imread(filename)	#CHANGE ME to the CV img
    width = im.shape[0]
    height = im.shape[1]
    if width < height:
      im = cv2.transpose(im)
      im = cv2.flip(im,1)

    imCopy = im
    imCopy = cv2.cvtColor(imCopy,cv2.COLOR_BGR2GRAY)
    imCopy = cv2.GaussianBlur(imCopy,(1,1),1000)
    imCopy = cv2.resize(imCopy, (426,426))
# imCopy = preprocess(imCopy)
    cv2.namedWindow('grayScale')
    cv2.moveWindow('grayScale',0,448)
    cv2.imshow('grayScale', imCopy)


    # Debug: uncomment to see registered images
    cardCount = 0
    try:
		for c in getCards(im):
		  card = find_closest_card(training,c)
		  c = cv2.resize(c,(426,426))
		  cv2.namedWindow((str(card)))
		  cv2.moveWindow((str(card)),0,0)
		  cv2.imshow(str(card),c)
#		  cv2.imshow(str(card),c)
		  cv2.waitKey(0)
		  cardCount+=1
    except ValueError:
	   print "cardCount: %d"%cardCount

    try:
        cards = [find_closest_card(training,c) for c in getCards(im,cardCount)]
        print cards
    except ValueError:
        #doNothing
		pass
		
  else:
      print __doc__
