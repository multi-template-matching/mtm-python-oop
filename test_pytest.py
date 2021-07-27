from skimage.data import coins
import MTM
from MTM.Detection import plotDetections
print( MTM.__version__ )
import numpy as np


#%% Get image and templates by cropping
image     = coins()
smallCoin = image[37:37+38, 80:80+41]
bigCoin   = image[14:14+59,302:302+65]

listLabels = ["small", "big"]
listTemplates = [smallCoin, bigCoin]

def test_simplest():
    return MTM.matchTemplates(image, 
                              listTemplates)

def test_searchRegion():
    return MTM.matchTemplates(image,
                              listTemplates,
                              searchBox=(0, 0, 300, 150))

def test_downscaling():
    return MTM.matchTemplates(image, 
                              listTemplates,
                              downscaling_factor=4)
    
if __name__ == "__main__":
    A = test_simplest()
    B = test_searchRegion()
    C = test_downscaling()
    print ("Number of hits:", len(A), len(B), len(C))