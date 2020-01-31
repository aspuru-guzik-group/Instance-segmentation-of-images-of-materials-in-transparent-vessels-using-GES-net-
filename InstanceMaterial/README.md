# Generator evaluator selector neural net for instance segmentation of materials in vessels.


## General
The net split region of the vessel in the image to the individual material phases, and assign class and score for each phase (Figure 1). This net runs as part of the hierarchical model that is described in the parent folder. Training is done on the [LabPics Dataset](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing)


## Running folder
Run in a hierarchical manner using several other nets. See the parent folder.


# Training


1. Download the LabPics data set from [Here](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)
2. Open the Train.py script
3. Set the path to the LabPics dataset’s main folder to the LabPicsFolder parameter.
4. Run the script 
5. Output trained model will appear in the /log subfolder or any folder set in Trained model Path.




## Net structure
The net is a [pointer net structure](https://arxiv.org/ftp/arxiv/papers/1902/1902.07810.pdf), a convolutional net that given an image a point in the image and ROI mask (Figure 1), return the segment region of the object instance containing the point within the ROI region, and a score that evaluates how good the predicted mask match the real phase region in the image in terms of IOU (intersection over union).


![](/InstanceMaterial/Figure1.png)
Figure 1
