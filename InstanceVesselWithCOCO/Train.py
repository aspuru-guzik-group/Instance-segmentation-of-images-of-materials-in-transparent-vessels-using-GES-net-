# Train GES /Pointer net for vessel and object instance/panoptic segmentation/evaluation and classification of segments as vessel/not vessel
#...............................Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import scipy.misc as misc
#import CocoPanoptic_Reader as Data_Reader
import COCO_Reader as COCO_Data_Reader
#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class
import LabPicsVesselInstanceReader
import Evaluator
##################################Input training data folders#########################################################################################
#.................................Main Input parametrs...........................................................................................
COCO_InpuDIr="/media/sagi/2T//DeleteLater/"
ImageDir="/media/sagi/2T/Data_zoo/CocoReloaded/train2017/"
MaskDir = {COCO_InpuDIr+"/All/"}
FullSegDir = COCO_InpuDIr+"/SegMapDir/"

LabPicsMainDir="/media/sagi/DefectiveHD/LabPicsV1.2/"
TrainDirLabPics=LabPicsMainDir+r"/Complex/Train/"
TestDirLabPics=LabPicsMainDir+r"/Complex/Test/"
#*******************************Other train parametes*********************************************************************************************************************
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
if not os.path.exists(TrainedModelWeightDir): os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""
Learning_Rate_Init=1e-5 # Initial learning rate
Learning_Rate=1e-5 # learning rate
#=========================Load weights====================================================================================================================
InitStep=0
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate_Init.npy"):
    Learning_Rate_Init=np.load(TrainedModelWeightDir+"/Learning_Rate_Init.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))
#...............Other training paramters..............................................................................
# Learning_Rate_Init=7e-6 # Initial learning rate
# Learning_Rate=7e-6 # learning rate

MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width


#Learning_Rate_Decay=Learning_Rate[0]/40 # Used for standart
Learning_Rate_Decay=Learning_Rate/20
StartLRDecayAfterSteps=200000
MaxPixels=340000*6#4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(100000010) # Max  number of training iteration

#-----------------Generate evaluator------------------------------------------------------------------------------------------------------------------------------------------------

Eval=Evaluator.Evaluator(TestDirLabPics,TrainedModelWeightDir+"/Evaluate.txt")
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net.AddEvaluationClassificationLayers(NumClass=1)
Net=Net.cuda()

if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))



optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch")

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
CoCoReader=COCO_Data_Reader.Reader(ImageDir,MaskDir,FullSegDir,NumClasses= 205, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ChemReader= LabPicsVesselInstanceReader .Reader(MainDir=TrainDirLabPics, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)

#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()

AVGLossSeg=-1# running average loss
AVGLossIOU=-1
AVGLossCat=-1

IOUConst=1 # Relative weight of  IOU score evalouator loss relative to the segmentation loss
CATConst=1# Relative weight of  classification  loss relative to the segmentation loss
#..............Start Training loop: Main Training...................................................................
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
    print(itr)
    if  np.random.rand() < 0.5:
          CoCoReader.ClassBalance = np.random.rand() < 0.5
          Imgs, SegmentMask, ROIMask, PointerMap,GTIsVessel  = CoCoReader.LoadBatch()
    else:
          Imgs, Ignore, SegmentMask, InstBG, ROIMask, PointerMap = ChemReader.LoadBatch()
          GTIsVessel = np.ones([Imgs.shape[0]])
    #--------------------------------------
    # for f in range(Imgs.shape[0]):
    #   #  if GTIsVessel[f]<1: continue
    #     print(GTIsVessel[f])
    #     Imgs[f, :, :, 0] *= 1-SegmentMask[f]
    #     Imgs[f, :, :, 1] *= ROIMask[f]
    #     misc.imshow(Imgs[f])
    #     misc.imshow((ROIMask[f] + SegmentMask[f] * 2 + PointerMap[f] * 3).astype(np.uint8)*40)
#------------------------------------------------------------------------------------------------------------------------------------------
    OneHotLabels = ConvertLabelToOneHotEncoding.LabelConvert(SegmentMask,
                                                             2)  # Convert labels map to one hot encoding pytorch
    # print("RUN PREDICITION")
    Prob, Lb, PredIOU, PredIsVessel = Net.forward(Images=Imgs, Pointer=PointerMap,
                                                   ROI=ROIMask)  # Run net inference and get prediction
    #  print(PredIOU)
    Net.zero_grad()
#............................Segmentation loss...........................................................................................
    LossSeg = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and ground truth label
#...................IOU score (evaluator) loss.............................................................................................
    Lb = Lb.data.cpu().numpy()
    Inter = (Lb * SegmentMask).sum(axis=1).sum(axis=1)
    Union = Lb.sum(axis=1).sum(axis=1) + SegmentMask.sum(axis=1).sum(axis=1) - Inter
    IOU = torch.autograd.Variable(torch.from_numpy((Inter / (Union + 0.000001)).astype(np.float32)).cuda(), requires_grad=False)
    LossIOU = (IOU - PredIOU[:, 0]).pow(2).mean()

# .............Loss classification loss...................................................................................
    GTcats = torch.autograd.Variable(torch.from_numpy(np.transpose(np.array([1 - GTIsVessel, GTIsVessel]).astype(np.float32))).cuda(), requires_grad=False)
    LossCats = -torch.mean(GTcats * torch.log((PredIsVessel  + 0.0000001)))


    # ..................................................Calculate general loss and backpropogate.........................................................................................

    Loss = LossIOU * IOUConst + LossSeg + LossCats * CATConst

    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight
    # ----------------Average loss--------------------------------------------------------------------------------------------------------------------------------------
    if AVGLossSeg == -1:
        AVGLossSeg = float(LossSeg.data.cpu().numpy())  # Calculate average loss for display
    else:
        AVGLossSeg = AVGLossSeg * 0.999 + 0.001 * float(LossSeg.data.cpu().numpy())  # Intiate runing average loss

    if AVGLossIOU == -1:
        AVGLossIOU = float(LossIOU.data.cpu().numpy())  # Calculate average loss for display
    else:
        AVGLossIOU = AVGLossIOU * 0.999 + 0.001 * float(LossIOU.data.cpu().numpy())  # Intiate runing average loss

    if AVGLossCat == -1:
        AVGLossCat = float(LossCats.data.cpu().numpy())  # Calculate average loss for display
    else:
        AVGLossCat = AVGLossCat * 0.999 + 0.001 * float(LossCats.data.cpu().numpy())  # Intiate runing average loss

    IOUConst = (AVGLossSeg / AVGLossIOU) * 0.23

    CATConst = (AVGLossSeg / AVGLossCat) * 0.04
    # --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 2000 == 0:  # and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in " + TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir + "/Learning_Rate.npy", Learning_Rate)
        np.save(TrainedModelWeightDir + "/Learning_Rate_Init.npy", Learning_Rate_Init)
        np.save(TrainedModelWeightDir + "/itr.npy", itr)
    if itr % 10000 == 0 and itr > 0:  # Save model weight once every 10k steps
        print("Saving Model to file in " + TrainedModelWeightDir + "/" + str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
    # ......................Write and display train loss..........................................................................
    if itr % 10 == 0:  # Display train loss

        txt = "\n" + str(itr) + "\t Seg Loss " + str(AVGLossSeg) + "\t IOU Loss " + str(
            AVGLossIOU) + "\t" + "\t Cat Loss " + str(AVGLossCat) + "\t" + str(Learning_Rate) + " Init_LR" + str(
            Learning_Rate_Init)
        print(txt)
        # Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
    # ----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr % 10000 == 0 and itr >= StartLRDecayAfterSteps:
        Learning_Rate -= Learning_Rate_Decay
        if Learning_Rate <= 1e-6:
            Learning_Rate_Init -= 1e-6
        if Learning_Rate_Init <= 1e-6:
                Learning_Rate_Init = 2e-6
        Learning_Rate = Learning_Rate_Init
        Learning_Rate_Decay = Learning_Rate / 20
        print("Learning Rate=" + str(Learning_Rate) + "   Learning_Rate_Init=" + str(Learning_Rate_Init))
        print(
            "======================================================================================================================")
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,
                                     weight_decay=Weight_Decay)  # Create adam optimizer
        torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
    # ----------------------------------------Evaluate-------------------------------------------------------------------------------------------

    if itr % 10000 == 0:
        print("Evaluating")
        Eval.Eval(Net, itr)


