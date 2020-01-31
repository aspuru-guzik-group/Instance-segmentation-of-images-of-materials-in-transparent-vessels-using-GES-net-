# Hierarchical instance/semantic segmentation of materials and vessels 


## General
Combination of three nets that act modular way to create semantic and instance aware segmentation of materials in a mostly transparent vessel for chemistry lab and other settings. Vessels correspond to (bottles cups, chemistry lab vessel, and other containers and glassware.) mostly transparent. The material phase corresponds to liquid solids, foams,powders, suspensions etc.


# Running prediction


1. Download the pre-trained system from [here]() or train the nets according to the instructions in the training section.
2. Set path to the folder of the input image to the: InputDir parameter.
3. Set path to the folder where the output annotations will be saved to the: OutDir parameter.
4. Run.py 


## Additional parameters:
If you trained the net yourself set paths to the trained model weight in:
SemanticNetTrainedModelPath,  InstanceVesselNetTrainedModelPath,  InstanceMaterialNetTrainedModelPath 
,  for semantic vessel and materials net models weights respectively

VesIOUthresh: Quality threshold for predicted vessel instance to be accepted.

MatIOUthresh: Quality threshold for predicted material instance to be accepted.

NumVessCycles: Number of attempts to search for vessel instance, increase the probability to find vessel but also running time

NumMatCycles: Number of attempts to search for material instance, increase the probability to find material phase but also running time

UseIsVessel: Only If the vessel instance net was trained with COCO, it could predict whether the instance belongs to a vessel or not, which can help to remove a false segment.




# Training
See each of the subfolders for instruction to training the individual nets


# Structure 


The hierarchical segmentation Figure 1, consist of 3 steps:
1. Semantic segmentation: finding the region of the vessels, fill level, liquid solids, and other phases (not instance aware).
2. Vessel instance segmentation: Splitting the region of the vessel region, into individual vessel instance regions.
3. Material instance segmentation: splitting the region of each vessel region into individual material phase instance regions.
# Nets 
## Semantic segmentation net
Semantic net is in the Semantic folder. 
A Fully convolutional net (FCN) PSP net that finds a binary segmentation map for each class. See the folder for more detail.


## Vessel instance net.
In folders: InstanceVessel and InstanceVesselWithCOCO, these are two training variations of the same net. One is trained with LabPics dataset and the other with LabPics and COCO, which make it more robust. This is a net that given an image, a point in the image, and ROI mask. The net returns the segment region of the instance containing the point within the ROI, a score that evaluates how good the predicted segment matches the real segment in terms of IOU, and also whether the instance corresponds to vessel or another thing (Figure 2). See the folder for more detail.




## Material instance net.
In folders: InstanceMaterial. This is a net that given an image, a point in the image, and ROI mask. The net returns the segment region of the material phase instance containing the point within the ROI region. A score that evaluates how good the predicted segment matches the real segment in terms of IOU, and also the class of the material phase (Figure 2). See the folder for more detail.








![](/Figure1.png)
Figure 1 : Hierarchical image segmentation


![](/Figure2.png)
Figure 1: GES net instance segmentation for materials and vessels


![](/Figure3.png)
Figure 3: Some results.
