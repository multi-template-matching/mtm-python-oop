"""
Multi -Template-Matching.

Implements object-detection with one or mulitple template images
Detected locations are represented as bounding boxes.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, feature, morphology, color
from .NMS import NMS
from .Detection import BoundingBox

__all__ = ['NMS']
__version__ = '1.5.4'

def findMaximas(corrMap, score_threshold=0.6, nObjects=float("inf")):
    """Get coordinates of the global (nnObjects=1) or local maximas with values above a threshold in the image of the correlation map."""
    # IF depending on the shape of the correlation map
    if corrMap.shape == (1,1): ## Template size = Image size -> Correlation map is a single digit representing the score
        listPeaks = np.array([[0,0]]) if corrMap[0,0]>=score_threshold else []

    else: # Correlation map is a 1D or 2D array
        nPeaks = 1 if nObjects==1 else float("inf") # global maxima detection if nObject=1 (find best hit of the score map)
        # otherwise local maxima detection (ie find all peaks), DONT LIMIT to nObjects, more than nObjects detections might be needed for NMS
        listPeaks = feature.peak_local_max(corrMap,
                                           threshold_abs=score_threshold,
                                           exclude_border=False,
                                           num_peaks=nPeaks).tolist()

    return listPeaks


def findMatches(image,
                listTemplates,
                listLabels=[],
                score_threshold=0.5,
                nObjects=float("inf"),
                searchBox=None):
    """
    Find all possible templates locations provided a list of template to search and an image.

    Parameters
    ----------
    - image  : Grayscale or RGB numpy array
              image in which to perform the search, it should be the same bitDepth and number of channels than the templates

    - listTemplates : list of templates as grayscale or RGB numpy array
                      templates to search in each image

    - nObjects: int
                expected number of objects in the image
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    - searchBox : tuple (X, Y, Width, Height) in pixel unit
                optional rectangular search region as a tuple

    Returns
    -------
    - list of hits encoded as [template index, score, (x,y,width, height)]
    """
    if nObjects!=float("inf") and type(nObjects)!=int:
        raise TypeError("nObjects must be an integer")

    elif nObjects<1:
        raise ValueError("At least one object should be expected in the image")

    ## Crop image to search region if provided
    if searchBox != None:
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset:yOffset+searchHeight, xOffset:xOffset+searchWidth]
    else:
        xOffset=yOffset=0

    listHit = []
    for index, template in enumerate(listTemplates):

        #print('\nSearch with template : ',templateName)

        corrMap   = feature.match_template(image, template)
        listPeaks = findMaximas(corrMap, score_threshold, nObjects)

        height, width = template.shape[0:2] # slicing make sure it works for RGB too
        label = listLabels[index] if listLabels else ""

        for peak in listPeaks :
            score = corrMap[tuple(peak)]
            bbox  = (int(peak[1]) + xOffset,
                     int(peak[0]) + yOffset,
                     width, height)

            hit = BoundingBox(bbox, score, index, label)
            listHit.append(hit) # append to list of potential hit before Non maxima suppression

    return listHit # All possible hits before Non-Maxima Supression


def matchTemplates(image,
                   listTemplates,
                   listLabels=[],
                   score_threshold=0.5,
                   maxOverlap=0.25,
                   nObjects=float("inf"),
                   searchBox=None):
    """
    Search each template in the image, and return the best nObjects location which offer the best score and which do not overlap.

    Parameters
    ----------
    - listTemplates : list of templates as 2D grayscale or RGB numpy array
                      templates to search in each image, associated to a label
    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates
    - nObjects: int
                expected number of objects in the image
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    - maxOverlap: float in range [0,1]
                This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
                If the ratio is over the maxOverlap, the lower score bounding box is discarded.
    - searchBox : tuple (X, Y, Width, Height) in pixel unit
                optional rectangular search region as a tuple

    Returns
    -------
    Pandas DataFrame with 1 row per hit and column "TemplateName"(string), "BBox":(X, Y, Width, Height), "Score":float
        if N=1, return the best matches independently of the score_threshold
        if N<inf, returns up to N best matches that passed the score_threshold
        if N=inf, returns all matches that passed the score_threshold
    """
    if maxOverlap<0 or maxOverlap>1:
        raise ValueError("Maximal overlap between bounding box is in range [0-1]")

    listHit  = findMatches(image, listTemplates, listLabels, score_threshold, nObjects, searchBox)
    bestHits = NMS(listHit, maxOverlap, nObjects)

    return bestHits


def drawBoxesOnRGB(image, listHit, thickness=2, colour=(255,0,0)):
    """
    Return a copy of the image with predicted template locations as bounding boxes overlaid on the image.
    TO DO:
        -Use a different color for every template index, need to draw sequentially for each template
        draw on a color mask first and then burn the mask ?

    Parameters
    ----------
    - image  : image in which the search was performed
    - tableHit: list of hit as returned by matchTemplates or findMatches
    - boxThickness: int
                    thickness of bounding box contour in pixels
    - boxColor: (int, int, int)
                RGB color for the bounding box
    - showLabel: Boolean
                Display label of the bounding box (field TemplateName)
    - labelColor: (int, int, int)
                RGB color for the label

    Returns
    -------
    outImage: RGB image
            original image with predicted template locations depicted as bounding boxes
    """
    # Convert Grayscale to RGB to be able to see the color bboxes
    if image.ndim == 2: outImage = color.gray2rgb(image) # convert to RGB to be able to show detections as color box on grayscale image
    else:               outImage = image.copy()

    palette = plt.cm.Set3.colors
    nColors = len(palette)

    mask = np.zeros(outImage.shape[:2], dtype="bool") # 1D boolean mask with True when bbox
    for hit in listHit:
        x,y,width,height = hit[1]
        template = hit[2]
        colorIndex = template % nColors # will return an integer in the range of palette
        rr, cc = draw.rectangle_perimeter(start=(y,x), extent=(height, width))
        mask[rr, cc] = True

    # Thicken rectangle contour
    if thickness>1:
        mask = morphology.dilation(mask, morphology.square(width=thickness))

    # Overlay rectangle on the image
    outImage[mask] = colour

    return outImage


def drawBoxesOnGray(image, listHit, thickness=2, gray=255):
    """
    Same as drawBoxesOnRGB but with Graylevel.

    If a RGB image is provided, the output image will be a grayscale image

    Parameters
    ----------
    - image  : image in which the search was performed
    - listHit: list of hit as returned by matchTemplates or findMatches
    - thickness: int
                thickness of bounding box contour in pixels
    - gray: int
                Gray level for the bounding box

    Returns
    -------
    outImage: Single channel grayscale image
            original image with predicted template locations depicted as bounding boxes
    """
    # Convert RGB to grayscale
    if image.ndim == 3: outImage = color.rgb2gray(image) # convert to RGB to be able to show detections as color box on grayscale image
    else:               outImage = image.copy()

    mask = np.zeros_like(outImage, dtype="bool")
    for hit in listHit:
        x,y,width,height = hit[1]
        rr, cc = draw.rectangle_perimeter(start=(y,x), extent=(height, width))
        mask[rr, cc] = True # boolean mask

    # Thicken rectangle contour
    if thickness>1:
        mask = morphology.dilation(mask, morphology.square(width=thickness))

    # Overlay rectangle on the image
    outImage[mask] = gray

    #if showLabel: cv2.putText(outImage, text=row['TemplateName'], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=labelScale, color=labelColor, lineType=cv2.LINE_AA)

    return outImage