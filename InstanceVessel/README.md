# Generator evaluator selector neural net for instance segmentation of vessels in an image (mostly transparent and glass vessels in a chemistry laboratory setting).


## General
The net split region of the vessels in the image to individual vessels (Figure 1). Vessels consist of general glass or transparent vessels, such as glass bottles and other glassware used in a lab (might also work on none transparent vessel). This net runs as part of the hierarchical model that is described in the parent folder. Training is done on the [LabPics Dataset](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing)


## Running
Run in a hierarchical manner using several other nets. See the parent folder for more details.


# Training


1. Download the LabPics data set from [Here](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)
2. Open the Train.py script
3. Set the path to the LabPics dataset’s main folder to the LabPicsFolder parameter.
4. Run the script 
5. Output trained model will appear in the /log subfolder or any folder set in Trained model Path.




## Net structure
The net is a [pointer net structure](https://arxiv.org/ftp/arxiv/papers/1902/1902.07810.pdf), a convolutional net that given an image a point in the image and ROI mask (Figure 1), return the segment region of the object instance containing the point within the ROI region, and a score that evaluates how good the predicted mask much the real object in the image in terms of IOU (intersection over union).


![](/InstanceVessel/Figure1.png)
Figure 1
