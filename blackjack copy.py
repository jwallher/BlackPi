from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
from time import sleep

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
    
    def processImage(image):
        pass

    def getCardsFromCamera(image):
        pass #Returns a list of card values

if __name__ == '__main__':
    # Make a new deck of cards
    initDeck()
    
    # Start the camera
    camera = PiCamera(resolution=(864, 864))
    #camera.resolution = (864, 864)
    rawCapture = PiRGBArray(camera)
    
    # Give the camera time to start up
    sleep(1)
    
    # DEBUG Pretend that we got these numbers from the camera
    count = 0
    cardSets = [ 
        ['A', '2'],
        ['3'],
        ['A'],
        ['10'],
        ['4'],
        ['3']
    ]
    
    try:
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
            image = rawCapture.array
            rawCapture.truncate(0)

            # Get value cards from camera
            # cards = getCardsFromCamera()
            
            # DEBUG
            cards = cardSets[count]
            count += 1

            for card in cards:
                addToHand(card)

            print 'I currently have %d' % getHandTotal()
            print "My chance of busting is about %.02f" % oddsToBust()

            if oddsToBust() < CONFIDENCE:
                print "\tHit Me!"
                hit = True
            else:
                print "\tI'll stand."
                hit = False
            
            printHand()
            #printDeck()
           
            """
            index = 1
            for card in hand:
                windowName = 'Card #%d' % index
                cv2.namedWindow(windowName)
                image = cv2.imread('games.jpg')
                cv2.imshow(windowName, image)
                cv2.resize(image, (250, 250) )
                cv2.moveWindow(windowName, 0 + 0, 100)
                index += 1

                cv2.waitKey(0)
                cv2.destroyAllWindows()
            """
            
            print 'Press Enter to comtinue...'
            raw_input()
        
        discardHand()
        camera.close()

    except KeyboardInterrupt:
        print 'Closing Program'
