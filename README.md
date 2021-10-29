[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/multi-template-matching/mtm-python-oop/main?filepath=tutorials)

# Multi-Template-Matching
Multi-Template-Matching is an accessible method to perform object-detection in images using one or several template images for the search.  
The strength of the method compared to previously available single-template matching, is that by combining the detections from multiple templates,
one can improve the range of detectable patterns. This helps if you expect variability of the object-perspective in your images, such as rotation, flipping...  
The detections from the different templates are not simply combined, they are filtered using Non-Maxima Supression (NMS) to prevent overlapping detections.  

The python implementations of mtm only perform the detection and filtering with NMS.  
For the templates, you can provde a list of images to use. You can also perform geometrical transformation (kind of data augmentation) of existing templates if you expect these transformation in the image (ex: rotation/flipping).  

This implementation relies on the packages scikit-image and shapely, but not on OpenCV contrary to the python implementation originally published (and still available).  
It is more object-oriented, especially it should be easier to adapt to other shapes (detection with rectangular template but outlining detected region with a non-rectangular shape), by implementing another type of Detection object.  
In this implementation the detection are of type `BoundingBox` which is derived from the shapely `Polygon` class. Therefore you can use all functions available for Polygon with a BoundingBox.  

The main function `mtm.matchTemplates` returns the best predicted locations provided a scoreThreshold and an optional number of objects expected in the image.  

The website of the project https://multi-template-matching.github.io/Multi-Template-Matching/ references most of the information, including presetnations, posters and recorded talks/tutorials.  

# Installation  
Coming soon.  

# Examples
Check out the [jupyter notebook tutorial](https://github.com/multi-template-matching/mtm-python-oop/tree/master/tutorials) for some example of how to use the package.  
You can run the tutorials online using Binder, no configuration needed ! (click the Binder banner on top of this page).  
To run the tutorials locally, install the package using pip as described above, then clone the repository and unzip it.  
Finally open a jupyter-notebook session in the unzipped folder to be able to open and execute the notebook.  
The [wiki](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji/wiki) section of this related repository also provides some information about the implementation.

# Citation
If you use this implementation for your research, please cite:
  
Thomas, L.S.V., Gehrig, J.  
_Multi-template matching: a versatile tool for object-localization in microscopy images_  
BMC Bioinformatics 21, 44 (2020). https://doi.org/10.1186/s12859-020-3363-7

# Related projects
See this [repo](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji) for the implementation as a Fiji plugin.  
[Here](https://nodepit.com/workflow/com.nodepit.space%2Flthomas%2Fpublic%2FMulti-Template%20Matching.knwf) for a KNIME workflow using Multi-Template-Matching.


# Origin of the work
This work has been part of the PhD project of **Laurent Thomas** under supervision of **Dr. Jochen Gehrig** at ACQUIFER.  

<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/Acquifer_Logo_60k_cmyk_300dpi.png" alt="ACQUIFER" width="400" height="80">     

# Funding
This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement No 721537 ImageInLife.  

<p float="left">
<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/ImageInlife.png" alt="ImageInLife" width="130" height="100">
<img src="https://github.com/multi-template-matching/MultiTemplateMatching-Python/blob/master/images/MarieCurie.jpg" alt="MarieCurie" width="130" height="130">
</p>
