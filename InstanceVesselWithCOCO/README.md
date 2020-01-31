# Generator evaluator selector neural net for instance segmentation of vessels in an image (mostly transparent and glass vessels in a chemistry laboratory setting) train with COCO and LabPics.


## General
The net finds vessel instances in the image  (as well as all other objects and none object region) in the image. Vessels consist of general glass or transparent vessels, such as glass bottles and other glassware used in a lab (might also work on none transparent objects). This net runs as part of the hierarchical model that is described in the parent folder. Training is done on a combination of the [LabPics Dataset](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) and the [COCO dataset](http://cocodataset.org/#download) 


## Running folder
Run in a hierarchical manner using several other nets. See the parent folder.


# Training


1. Download the LabPics data set from [Here](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)


2. Download the [COCO panoptic dataset](http://cocodataset.org/#download) annotation and train images.
### Converting COCO dataset into training data
3. Open script GenerateTraininigDataEqualClassForPointer/RunGenerateData.py
4. Set the COCO dataset image folder to the ImageDir parameter.
5. Set the COCO panoptic annotation folder to the AnnotationDir parameter.
6. Set the COCO panoptic .json file to the DataFile parameter.
7. Set the output folder (where the generated data will be saved) to the OutDir parameter.
8. Run script. 
### Training
9. Open the Train.py script
10. Set the path to the LabPics dataset main folder to the LabPicsMainDir parameters.
11. Set the path to the COCO generated data (OutDir, step 7)  to the COCO_InpuDIr parameter.
12. Set the COCO image dir to ImageDir.
13. Run the script 
14. Output trained model will appear in the /log subfolder or any folder set in Trained model Path












## Net structure
The net is a [pointer net structure](https://arxiv.org/ftp/arxiv/papers/1902/1902.07810.pdf), a convolutional net that given an image a point in the image and ROI mask (Figure 1), return the segment region of the object instance containing the point within the ROI region, and a score that evaluates how good the predicted mask much the real object in the image in terms of IOU (intersection over union). In addition, the net predicts whether the instance corresponds to a vessel or non-vessel region.


![](/InstanceVesselWithCOCO/Figure1.png)
Figure 1
