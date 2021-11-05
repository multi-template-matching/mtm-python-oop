[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/multi-template-matching/mtm-python-oop/main?filepath=tutorials)

# Multi-Template-Matching (mtm) : the object-oriented python implementation  
Multi-Template-Matching is an accessible method to perform object-detection in images using one or several template images for the search.  
The strength of the method compared to previously available single-template matching, is that by combining the detections from multiple templates,
one can improve the range of detectable patterns. This helps if you expect variability of the object-perspective in your images, such as rotation, flipping...  
The detections from the different templates are not simply combined, they are filtered using Non-Maxima Suppression (NMS) to prevent overlapping detections.  

The python implementations of mtm only perform the detection and filtering with NMS.  
For the templates, you can provide a list of images to use. You can also perform geometrical transformation (kind of data augmentation) of existing templates if you expect these transformation in the image (ex: rotation/flipping).  

This implementation relies on the packages scikit-image and shapely, but not on OpenCV contrary to the python implementation originally published (and still available).  
It is more object-oriented, especially it should be easier to adapt to other shapes (detection with rectangular template but outlining detected region with a non-rectangular shape), by implementing another type of Detection object.  
In this python implementation, the detections are of type `BoundingBox` and hold a reference to a shapely `Polygon` object (a subtype of [geometric object](https://shapely.readthedocs.io/en/latest/manual.html#geometric-objects)).  
While most functions required for multi-template-matching are directly available through the BoundingBox object, you can also use functions and attributes available to shapely Polygon by accessing the `polygon`attribute of a BoundingBox.  

Core functions available in mtm are : 

- the main function `mtm.matchTemplates`  
It returns the best predicted locations provided a scoreThreshold and an optional number of objects expected in the image.  
It performs the search for each template followed by overlap-based Non-Maxima Suppression (NMS) to remove redundant/overlapping detections.  
If a number N of expected object is mentioned, it returns at max N detection but potentially less depending on the score threshold.  

- the function `mtm.findMatches`  
It performs the search for each template and return all detections above the score-threshold, or a single top-score detection for each template if `singleMatch` is true.  
Contrary to `mtm.matchTemplates`, __it does not perform NMS__ so you will potentially get overlapping detections.  
Usually one should use directly `mtm.matchTemplates`.  

The website of the project https://multi-template-matching.github.io/Multi-Template-Matching/ references most of the information, including presentations, posters and recorded talks/tutorials.  
The [wiki](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji/wiki) section of this related repository also provides some information about the implementation.  

# Installation  
Open a command prompt (or Anaconda prompt if using Anaconda) and type  
`pip install mtm` 

For development purpose, you can clone/download this repo, open a command prompt in the root directory of the repo and use pip to install the package in editable mode.  
`pip install -e .`  
mind the dot specifying to use the active directory (ie the one you open the prompt in).  
In editable mode, any change to the source code is directly reflected the next time you import the package.  

# Examples
Check out the [jupyter notebook tutorial](https://github.com/multi-template-matching/mtm-python-oop/tree/master/tutorials) for some example of how to use the package.  
You can run the tutorials online using Binder, no configuration needed ! (click the Binder banner on top of this page).  

To run the tutorials locally, install the package using pip as described above, then clone/download the repository and unzip it.  
Finally open a jupyter-notebook session in the unzipped folder to be able to open and execute the notebooks.  

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
