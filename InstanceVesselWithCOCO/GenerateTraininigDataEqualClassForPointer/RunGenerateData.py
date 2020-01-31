# Convert COCO panoptic to Training data for GES/pointer net

############################################################################################################################################################################################
import  DataGeneratorForPointerSegmentation
##############################################Input Coco folder#################################################################################################################################
ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/" # image folder (coco training) train set
AnnotationDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_train2017/" # annotation maps from coco panoptic train set
DataFile="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_train2017.json" # Json Data file coco panoptic train set
OutDir="/scratch/gobi2/seppel/ConvertedCOCO/"
##########################################Generate data###############################################################################################################################33

x=DataGeneratorForPointerSegmentation.Generator(ImageDir,AnnotationDir,OutDir, DataFile)
x.Generate()

