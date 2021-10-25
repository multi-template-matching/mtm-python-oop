"""
Make sure the active directory is the directory of the repo when running the test in a IDE.
"""
from skimage.data import coins
import mtm
from mtm.detection import plotDetections
print( mtm.__version__ )
import numpy as np

#%% Get image and templates by cropping
image     = coins()
smallCoin = image[37:37+38, 80:80+41]
bigCoin   = image[14:14+59,302:302+65]

asHigh  = image[:,10:50]
asLarge = image[50:70,:]

listLabels = ["small", "big"]
listTemplates = [smallCoin, bigCoin]


#%% Perform matching
listHit      = mtm.findMatches(image, listTemplates, listLabels)
singleObject = mtm.findMatches(image, listTemplates, listLabels, nObjects=1)  # there should be 1 top hit per template

finalHits = mtm.matchTemplates(image,
                               listTemplates,
                               listLabels,
                               scoreThreshold=0.4,
                               maxOverlap=0)

print("Found {} coins".format(len(finalHits)))
print (np.array(finalHits)) # better formatting with array

#%% Display matches
plotDetections(image, 
               finalHits, 
               showLegend=True, 
               showScore=True)