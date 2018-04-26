from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
from time import sleep
import sys
import numpy as np
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages/") 

values = ['A','1','2','3','4','5','6','7','8','9','10','J','Q','K']
deck = [] # Make an empty set of cards
hand = []

CONFIDENCE = 50.00

def initDeck():
    for suit in range(4):
        for value in values:
            deck.append(value)
def printDeck():
    print 'Deck:'
    for card in deck:
        print '\t%s' % card

def addToHand(card):
    hand.append(card)
    deck.remove(card)

def discardHand():
    hand[:] = []

def printHand():
    print 'Hand:'
    for card in hand:
        print '\t%s' % card

def getHandTotal():
    total = 0
    for card in hand:
        try:
            total += int(card)
            #print 'Added %d' % int(card)
        except ValueError:
            #print 'Face Card Encountered!'

            if card == 'A':
                total += 11
            else:
                total += 10

    if total > 21:
        aces = hand.count('A')
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
     
    return total

def oddsToBust():
    total = getHandTotal()
    bustNumber = 22 - total #The number that will put you over the edge
    safeCards = 0
    
    #print total
    #print bustNumber

    if total <= 11:
        return float(0)

    if total > 21:
        return float(100)
    
    for safeNumber in range(1, bustNumber):
        #if safeNumber == 1 or safeNumber == 11:
        if safeNumber == 1:
            #print 'A'
            safeCards += deck.count('A')

        if safeNumber == 10:
            #print '10'
            safeCards += deck.count('10')
            safeCards += deck.count('J')
            safeCards += deck.count('Q')
            safeCards += deck.count('K')
        
        if safeNumber in range(2, 10):
            #print safeNumber
            safeCards += deck.count(str(safeNumber))
    
    return 100.00 - ((float(safeCards) / float(len(deck))) * 100)


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
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
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


def get_training(training_labels_filename,training_image_filename,num_training_cards,avoid_cards=None):
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
  return training

if __name__ == '__main__':
	# Make a new deck of cards
    initDeck()
    
    # Start the camera
    camera = PiCamera(resolution=(864, 864))
    #camera.resolution = (864, 864)
    rawCapture = PiRGBArray(camera)
    
    # Give the camera time to start up
    sleep(1)
    
    num_cards=30
    training_image_filename='train2.png'
    training_labels_filename='train2.tsv'
    num_training_cards=56
    training = get_training(training_labels_filename,training_image_filename,num_training_cards)

    hit = True
    while(hit):
        # Capture the picture in the right format
        #if camera is open:
        camera.start_preview()
        print 'Press Enter to capture a photo.'
        raw_input()
        #camera.capture(rawCapture, format="bgr", resize=(864,864))
        camera.capture(rawCapture, format="bgr")
        camera.stop_preview()
        im = rawCapture.array
        rawCapture.truncate(0)

        # Get value cards from camera
        # cards = getCardsFromCamera()

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
			for i,c in enumerate(getCards(im,num_cards)):
		  		card = find_closest_card(training,c,)
		  		c = cv2.resize(c,(426,426))
				cv2.namedWindow((str(card)))
				cv2.moveWindow((str(card)),0,0)
				cv2.imshow(str(card),c)
				#cv2.imshow(str(card),c)
				cv2.waitKey(0)
				cardCount+=1
    	except ValueError:
	   		print "cardCount: %d"%cardCount

    	try:
        	cards = [find_closest_card(training,c) for c in getCards(im,cardCount)]
        	print cards
        	for card in cards:
        		card = [0:]

        	print "Modified Cards:"
        	for card in cards:
        		print card
        		
        	#todo: adjust cards from list of tuples to list of strings
    	except ValueError:
        	#doNothing
			pass

		#todo: fix cards
		print 'I currently have %d' % getHandTotal()
        print "My chance of busting is about %.02f" % oddsToBust()

        if oddsToBust() < CONFIDENCE:
            print "\tHit Me!"
            hit = True
        else:
            print "\tI'll stand."
            hit = False

		discardHand()
        	camera.close()

