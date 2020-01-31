#Train GES net for instance segmentation  of vessels
#...............................Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
#import scipy.misc as misc

import FCN_NetModel as NET_FCN # The net Class
import LabPicsVesselInstanceReader as LabPicsInstanceReader
import Evaluator
##################################Input paramaters #########################################################################################
LabPicsFolder="/scratch/gobi2/seppel/LabPicsV1.2/"
TrainDirLabPics=LabPicsFolder+r"/Complex/Train//"
TestDirLabPics=LabPicsFolder+r"/Complex/Test//"
#*************************************Initial Train Parameter***************************************************************************************************************
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

#-----------------Generate evaluator for net accuracy------------------------------------------------------------------------------------------------------------------------------------------------

Eval=Evaluator.Evaluator(TestDirLabPics,TrainedModelWeightDir+"/Evaluate.txt")
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load main net
Net.AddEvaluationClassificationLayers(NumClass=1) # Add Evaluation and classification layers
Net=Net.cuda()

if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))



optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch")

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ChemReader= LabPicsInstanceReader.Reader(MainDir=TrainDirLabPics, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)

#--------------------------- Create logs files for saving loss statitics during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()

AVGLossSeg=-1# running average loss
AVGLossIOU=-1
AVGLossCat=-1

IOUConst=1 # Weight ratio between evaluator and generator (segmentation) losses
CATConst=1
#..............Start Training loop: Main Training...................................................................
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
    print(itr)
    Imgs, Ignore, SegmentMask, InstBG, ROIMask, PointerMap = ChemReader.LoadBatch()
    GTIsVessel = np.ones([Imgs.shape[0]])
    #---------------------display loaded data-----------------
    # for f in range(Imgs.shape[0]):
    #     Imgs[f, :, :, 0] *= 1-SegmentMask[f]
    #     Imgs[f, :, :, 1] *= ROIMask[f]
    #     misc.imshow(Imgs[f])
    #     misc.imshow((ROIMask[f] + SegmentMask[f] * 2 + PointerMap[f] * 3).astype(np.uint8)*40)
#-----------------------------------Calculate segmentation loss-------------------------------------------------------------------------------------------------------
    OneHotLabels = ConvertLabelToOneHotEncoding.LabelConvert(SegmentMask,2)  # Convert labels map to one hot encoding pytorch
    # print("RUN PREDICITION")
    Prob, Lb, PredIOU, PredIsVessel = Net.forward(Images=Imgs, Pointer=PointerMap,ROI=ROIMask)  # Run net inference and get prediction
    #  print(PredIOU)
    Net.zero_grad()
    LossSeg = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate segmentation loss between prediction and ground truth label
#----------------------calculate evaluator IOU prediction loss---------------------------------------------------------------------------------
    Lb = Lb.data.cpu().numpy()
    Inter = (Lb * SegmentMask).sum(axis=1).sum(axis=1)
    Union = Lb.sum(axis=1).sum(axis=1) + SegmentMask.sum(axis=1).sum(axis=1) - Inter
    IOU = torch.autograd.Variable(torch.from_numpy((Inter / (Union + 0.000001)).astype(np.float32)).cuda(), requires_grad=False)
    LossIOU = (IOU - PredIOU[:, 0]).pow(2).mean()

# .............classification loss (not used in this case)...................................................................................
#     GTcats = torch.autograd.Variable(
#     torch.from_numpy(np.transpose(np.array([1 - GTIsVessel, GTIsVessel]).astype(np.float32))).cuda(), requires_grad=False)
#     LossCats = -torch.mean(GTcats * torch.log((PredIsVessel  + 0.0000001)))


    # ................................Calculate general loss...........................................................................................................

    Loss = LossIOU * IOUConst + LossSeg# + LossCats * CATConst # Unify evaluation and segmentation lossed

    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight
    # ----------------Average loss statistics--------------------------------------------------------------------------------------------------------------------------------------
    if AVGLossSeg == -1:
        AVGLossSeg = float(LossSeg.data.cpu().numpy())  # Calculate average loss for display
    else:
        AVGLossSeg = AVGLossSeg * 0.999 + 0.001 * float(LossSeg.data.cpu().numpy())  # Intiate runing average loss

    if AVGLossIOU == -1:
        AVGLossIOU = float(LossIOU.data.cpu().numpy())  # Calculate average loss for display
    else:
        AVGLossIOU = AVGLossIOU * 0.999 + 0.001 * float(LossIOU.data.cpu().numpy())  # Intiate runing average loss

    # if AVGLossCat == -1:
    #     AVGLossCat = float(LossCats.data.cpu().numpy())  # Calculate average loss for display
    # else:
    #     AVGLossCat = AVGLossCat * 0.999 + 0.001 * float(LossCats.data.cpu().numpy())  # Intiate runing average loss

    IOUConst = (AVGLossSeg / AVGLossIOU) * 0.05

  #  CATConst = (AVGLossSeg / AVGLossCat) * 0.1
    ################################################################################################################################################################################33
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
    if itr % 40 == 0:  # Display train loss

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
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer
        torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
    # ----------------------------------------Evaluate-------------------------------------------------------------------------------------------

    if itr % 10000 == 0:
        print("Evaluating")
        Eval.Eval(Net, itr)


