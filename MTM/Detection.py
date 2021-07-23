"""
Generic Detection object with functions to compute intersection, union and IoU between detections.

This class should be generic thanks to getter functions and not specific to a specific implementation
Basically any Detection that fullfills the methods below should work
Good for skimage not for opencv NMSBoxes
"""
from shapely.geometry import polygon

class BoundingBox(polygon.Polygon):
    """
    Describe a detection as a rectangular bounding box.

    Parameters
    ----------
    bbox, tuple of 4 ints:
        x,y, width, height dimensions of the rectangle outlining the detection with x,y the top left corner

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
