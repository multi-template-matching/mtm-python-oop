"""
Multi-Template-Matching.

Implements object-detection with one or mulitple template images
Detected locations are represented as bounding boxes.
"""
import numpy as np
from skimage import feature, transform
from . import nms
from . import detection

__version__ = '1.0.1'


def findMaximas(corrMap, score_threshold=0.6, nObjects=float("inf")):
    """
    Maxima detection in correlation map.
    
    Get coordinates of the global (nObjects=1)
    or local maximas with values above a threshold
    in the image of the correlation map.
    """
    # IF depending on the shape of the correlation map
    if corrMap.shape == (1, 1):  # Template size = Image size -> Correlation map is a single digit representing the score
        listPeaks = np.array([[0, 0]]) if corrMap[0, 0] >= score_threshold else []

    else:  # Correlation map is a 1D or 2D array
        nPeaks = 1 if nObjects == 1 else float("inf")  # global maxima detection if nObject=1 (find best hit of the score map)
        # otherwise local maxima detection (ie find all peaks), DONT LIMIT to nObjects, more than nObjects detections might be needed for NMS
        listPeaks = feature.peak_local_max(corrMap,
                                           threshold_abs=score_threshold,
                                           exclude_border=False,
                                           num_peaks=nPeaks).tolist()

    return listPeaks


def findMatches(image,
                listTemplates,
                listLabels=None,
                score_threshold=0.5,
                nObjects=float("inf"),
                searchBox=None,
                downscaling_factor=1):
    """
    Find all possible templates locations provided a list of template to search and an image.
    
    Parameters
    ----------
    - image  : Grayscale or RGB numpy array
              image in which to perform the search, it should be the same bitDepth and number of channels than the templates
    
    - listTemplates : list of templates as grayscale or RGB numpy array
                      templates to search in each image
    
    - listLabels (optional) : list of string labels associated to the templates (order must match the templates in listTemplates).
                              these labels can describe categories associated to the templates
                              
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
   
    - nObjects: 1 or infinity (default)
                If N=1 use global maxima detection, otherwise use local maxima detection
   

    - searchBox (optional): tuple (x y, width, height) in pixels
                limit the search to a rectangular sub-region of the image
    
    - downscaling_factor: int >= 1, default 1 (ie no downscaling)
               speed up the search by downscaling the template and image before running the template matching.
               Detected regions are then rescaled to original image sizes.
               
    Returns
    -------
    - List of BoundingBoxes
    """
    if nObjects != float("inf") and type(nObjects) != int:
        raise TypeError("nObjects must be an integer")

    if nObjects < 1:
        raise ValueError("At least one object should be expected in the image")

    if (listLabels is not None and
       (len(listTemplates) != len(listLabels))):
        raise ValueError("len(listTemplate) != len(listLabels).\nIf listLabels is provided, there must be one label per template.")

    if downscaling_factor < 1:
        raise ValueError("Downscaling factor must be >= 1")

    # Crop image to search region if provided
    if searchBox is not None:
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset:yOffset+searchHeight,
                      xOffset:xOffset+searchWidth]
    else:
        xOffset = yOffset = 0
    
    # Check template smaller than image (or search region)
    for index, template in enumerate(listTemplates):
        
        templateSmallerThanImage = all(templateDim <= imageDim for templateDim, imageDim in zip(template.shape, image.shape))
        
        if not templateSmallerThanImage :
            fitIn = "searchBox" if (searchBox is not None) else "image"
            raise ValueError("Template '{}' at index {} in the list of templates is larger than {}.".format(template, index, fitIn) )
    
    # make a downscaled copy of the image if downscalingFactor != 1
    if downscaling_factor != 1: # dont use anti-aliasing to keep small structure and faster
        image = transform.rescale(image, 1/downscaling_factor, anti_aliasing = False)

    listHit = []
    for index, template in enumerate(listTemplates):
        
        if downscaling_factor != 1:  # make a downscaled copy of the current template
            template = transform.rescale(template, 1/downscaling_factor, anti_aliasing = False)
            
        corrMap = feature.match_template(image, template)
        listPeaks = findMaximas(corrMap, score_threshold, nObjects)

        height, width = template.shape[0:2]  # slicing make sure it works for RGB too
        label = listLabels[index] if listLabels else ""

        for peak in listPeaks:
            score = corrMap[tuple(peak)]
            
            # bounding-box dimensions
            # resized to the original image size (hence x downscaling factor)
            xy = np.array(peak[::-1]) * downscaling_factor + (xOffset, yOffset) # -1 since peak is in (i, j) while we want (x,y) coordinates
            bbox = tuple(xy) + (width  * downscaling_factor, 
                                height * downscaling_factor) # in theory we could use original template width/height before downscaling, but using the size of the actually used template is more correct 

            hit = detection.BoundingBox(bbox, score, index, label)
            listHit.append(hit)  # append to list of potential hit before Non maxima suppression

    return listHit  # All possible hits before Non-Maxima Supression


def matchTemplates(image,
                   listTemplates,
                   listLabels=None,
                   score_threshold=0.5,
                   maxOverlap=0.25,
                   nObjects=float("inf"),
                   searchBox=None,
                   downscaling_factor=1):
    """
    Search each template in the image, and return the best nObjects locations which offer the best score and which do not overlap.
   
    Parameters
    ----------
    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates
               
    - listTemplates : list of templates as 2D grayscale or RGB numpy array
                      templates to search in each image
    
    - listLabels (optional) : list of strings
                              labels, associated the templates. The order of the label must match the order of the templates in listTemplates.
    
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    
    - maxOverlap: float in range [0,1]
                This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
                If the ratio is over the maxOverlap, the lower score bounding box is discarded.
    
    - nObjects: int
               expected number of objects in the image
    
    - searchBox : tuple (x y, width, height) in pixels
                limit the search to a rectangular sub-region of the image
                
    - downscaling_factor: int >= 1, default 1 ie no downscaling
               speed up the search by downscaling the template and image before running the template matching.
               Detected regions are then rescaled to original image sizes.
               
    Returns
    -------
    List of BoundingBoxes
        if nObjects=1, return the best BoundingBox independently of the score_threshold and maxOverlap
        if nObjects<inf, returns up to N best BoundingBoxes that passed the score_threshold and Non-Maxima Suppression
        if nObjects='inf'(string), returns all BoundingBoxes that passed the score_threshold and Non-Maxima Suppression
        
    """
    if maxOverlap<0 or maxOverlap>1:
        raise ValueError("Maximal overlap between bounding box is in range [0-1]")

    listHit  = findMatches(image, listTemplates, listLabels, score_threshold, nObjects, searchBox, downscaling_factor)
    bestHits = nms.runNMS(listHit, maxOverlap, nObjects)

    return bestHits     