"""
Generic Detection object with functions to compute intersection, union and IoU between detections.

This class should be generic thanks to getter functions and not specific to a specific implementation
Basically any Detection that fullfills the methods below should work
Good for skimage not for opencv NMSBoxes
"""
from shapely.geometry import polygon
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

def plotDetections(image, listDetections, thickness=2, showLegend=False, showScore=False):
    """
    Plot the detections overlaid on the image.
   
    This generates a Matplotlib figure and displays it.
    Detections with identical template index (ie categories)
    are shown with identical colors.
    
    The figure can be further costumised after calling this function with following matplotlib.pyplot calls.
    
    Parameters
    ----------
    - image  :
        image in which the search was performed
    
    - listDetections:
        list of detections as returned by matchTemplates or findMatches
    
    - thickness (optional, default=2): int
        thickness of plotted contour in pixels
    
    - showLegend (optional, default=False): Boolean
        Display a legend panel with the category labels for each color.
        This works if the Detections have a label
        (not just "", in which case the legend is not shown).
    
    - showScore (optional, default=False): Boolean
        Display the score of the corresponding hit next to a plotted contour.
    """
    plt.figure()
    plt.imshow(image, cmap="gray")  # cmap gray only impacts gray images
    # RGB are still displayed as color

    # Load a color palette for categorical coloring of detections
    # ie same category (identical tempalte index) = same color
    palette = plt.cm.Set3.colors
    nColors = len(palette)

    if showLegend:
        mapLabelColor = {}

    for detection in listDetections:

        # Get color for this category
        colorIndex = detection.get_template_index() % nColors  # will return an integer in the range of palette
        color = palette[colorIndex]

        plt.plot(*detection.get_lists_xy(),
                 linewidth=thickness,
                 color=color)

        if showScore:
            (x, y, width, height) = detection.get_xywh()
            plt.annotate(round(detection.get_score(), 2),
                         (x + width/3, y + height/3),
                         ha="center",
						 fontsize=height/4)

        # If show legend, get detection label and current color
        if showLegend:

            label = detection.get_label()

            if label != "":
                mapLabelColor[label] = color

    # Finally add the legend if mapLabelColor is not empty
    if showLegend :

        if not mapLabelColor:  # Empty label mapping
            warnings.warn("No label associated to the templates." +
                          "Skipping legend.")

        else:  # meaning mapLabelColor is not empty

            legendLabels = []
            legendEntries = []

            for label, color in mapLabelColor.items():
                legendLabels.append(label)
                legendEntries.append(Line2D([0], [0], color=color, lw=4))

            plt.legend(legendEntries, legendLabels)


class BoundingBox(polygon.Polygon):
    """
    Describe a detection as a rectangular axis-aligned bounding box.

    Parameters
    ----------
    bbox, tuple of 4 ints:
        x, y, width, height dimensions of the rectangle outlining the detection with x,y the top left corner

    score, float:
        detection score

    template_index, int (optional)
        positional index of the template in the iniial list of templates

    label, string (optional)
        label for the detection (e.g. a category name or template name)
    """

    def __init__(self, bbox, score, template_index=0, label=""):
        x, y, width, height = bbox
        super().__init__( [(x,y), (x+width-1,y), (x+width-1, y+height-1), (x, y+height-1)] )
        self.xywh = bbox
        self.score = score
        self.template_index = template_index
        self.label = label

    def get_label(self):
        """Return the label associated to this detection (ex a category)."""
        return self.label

    def get_score(self):
        return self.score

    def get_template_index(self):
        """
        Return the positional index of the template
        associated to this detection in the original list of templates.
        """
        return self.template_index

    def __str__(self):
        name = "(BoundingBox, score:{:.2f}, xyxy:{}, index:{}".format(self.get_score(),
                                          tuple(map(int, self.bounds)),
                                          self.get_template_index()
                                          )

        label = self.get_label()
        if label:
            name+= ", " + label
        name += ")"

        return name

    def get_xywh(self):
        """Return the bounding-box dimensions as xywh. """
        return self.xywh

    def __repr__(self):
        return self.__str__()

    def intersection_area(self, detection2):
        """Compute the interesection area between this bounding-box and another detection (bounding-box or other shape)."""
        return self.intersection(detection2).area

    def union_area(self, detection2):
        """Compute the union area between this bounding-box and another detection (bounding-box or other shape)."""
        return self.union(detection2).area

    def intersection_over_union(self, detection2):
        """
        Compute the ratio intersection/union (IoU) between this bounding-box and another detection (bounding-box or other shape).
        The IoU is 1 is the shape fully overlap (ie identical sizes and positions).
        It is 0 if they dont overlap.
        """
        return self.intersection_area(detection2)/self.union_area(detection2)

    def get_lists_xy(self):
        """
        Return a tuple of 2 arrays for x and y coordinates.

        The lists correspond to the coordinates
        for the corners of the detection shape.
        """
        return self.exterior.xy
    
    @staticmethod
    def rescale_bounding_boxes(listDetectionsDownscaled, downscaling_factor):
        """
        Rescale detected bounding boxes to the original image resolution, when downscaling was used for the detection.
        
        Parameters
        ----------
        - listDetections : list of BoundingBox items
            List with 1 element per hit and each element containing "Score"(float), "BBox"(X, Y, X, Y), "Template_index"(int), "Label"(string)
        
        - downscaling_factor: int >= 1
                   allows to rescale by multiplying coordinates by the factor they were downscaled by
        Returns
        -------
        listDetectionsupscaled : list of BoundingBox items
            List with 1 element per hit and each element containing "Score"(float), "BBox"(X, Y, X, Y) (in coordinates of the full scale image), "Template_index"(int), "Label"(string)
        """
        listDetectionsUpscaled = []
    
        for detection in listDetectionsDownscaled:
            
            # Compute rescaled coordinates 
            xywh_upscaled = [coordinate * downscaling_factor for coordinate in detection.get_xywh() ]
    
            detectionUpscaled = BoundingBox(xywh_upscaled, 
                                            detection.get_score(), 
                                            detection.get_template_index(), 
                                            detection.get_label())
    
            listDetectionsUpscaled.append(detectionUpscaled)
    
        return listDetectionsUpscaled       

    """
    # If you were using a different implementation for Detection objects
    # These 2 functions should also be implemented
    # Here they are inherited from shapely's Polygon
    def overlaps(self, detection2):
    def contains(self, detection2)
    """


if __name__ == "__main__":
    detection = BoundingBox((0, 0, 10, 10), 0.5, label="Test")
    nolabel = BoundingBox((0, 0, 10, 10), 0.5)

    print(detection)
    print(nolabel)
    print([detection, nolabel])
