# TTrain GES pointer net to predict instances of materials phases the instance material class and the segmentation quality (IOU)
#...............................Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import Evaluator
import scipy.misc as misc

#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class
import LabPicsMaterialInstanceReader as LabPicsInstanceReader
import torch
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
LabPicsDir="/scratch/gobi2/seppel/LabPicsV1.2"
TrainDirLabPics=LabPicsDir+r"/Complex/Train/"
TestDirLabPics=LabPicsDir+r"//Complex/Test/"
#********************************************************************************************************************************************

#****************************************************************************************************************************************************
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
if not os.path.exists(TrainedModelWeightDir):
    os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""
Learning_Rate_Init=1e-5 # Initial learning rate
Learning_Rate=1e-5 # learning rate

#-----------------Generate evaluator------------------------------------------------------------------------------------------------------------------------------------------------

Eval=Evaluator.Evaluator(TestDirLabPics,TrainedModelWeightDir+"/Evaluat.txt")
#=========================Load net weights====================================================================================================================
InitStep=1
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
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained


if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
Net.AddEvaluationClassificationLayers(NumClass=20)
Net=Net.cuda()


optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch")

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ChemReader= LabPicsInstanceReader.Reader(MainDir=TrainDirLabPics, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)

#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#..............Start Training loop: Main Training....................................................................

AVGLossSeg=-1# running average loss
AVGLossIOU=-1
AVGLossCat=-1

IOUConst=1
CATConst=1
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
    print(itr)
    ChemReader.ClassBalance=np.random.rand()<0.3 # Read cl
    Imgs, Ignore, SegmentMask, InstBG, ROIMask, PointerMap, CatList = ChemReader.LoadBatch()
    #--------------------------------------
    # for f in range(Imgs.shape[0]):
    #     Imgs[f, :, :, 0] *= 1-SegmentMask[f]
    #     Imgs[f, :, :, 1] *= ROIMask[f]
    #     misc.imshow(Imgs[f])
    #     print(CatList[f])
    #     misc.imshow((ROIMask[f] + SegmentMask[f] * 2 + PointerMap[f] * 3).astype(np.uint8)*40)
    # print(ROIMask.shape)
    #----------------------------------------------
    OneHotLabels = ConvertLabelToOneHotEncoding.LabelConvert(SegmentMask, 2) #Convert labels map to one hot encoding pytorch
    #print("RUN PREDICITION")
    Prob, Lb, PredIOU, Predclasslist=Net.forward(Images=Imgs,Pointer=PointerMap,ROI=ROIMask) # Run net inference and get prediction
  #  print(PredIOU)
    Net.zero_grad()
#--------------Segmentation loss (generator-----------------------------------
    LossSeg = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and ground truth label

#------------IOU evaluation loss-------------------------------------------------
    Lb=Lb.data.cpu().numpy()
    Inter=(Lb*SegmentMask).sum(axis=1).sum(axis=1)
    Union=Lb.sum(axis=1).sum(axis=1)+SegmentMask.sum(axis=1).sum(axis=1)-Inter
    IOU = torch.autograd.Variable(torch.from_numpy((Inter / (Union+0.000001)).astype(np.float32)).cuda(), requires_grad=False)
    LossIOU=(IOU-PredIOU[:,0]).pow(2).mean()

 #.............Classification Loss...................................................................................
    GTcats=torch.autograd.Variable(torch.from_numpy(np.transpose(np.array([1 - CatList, CatList]).astype(np.float32))).cuda(),requires_grad=False)
    for c in range(len(Predclasslist)):
        if c>=GTcats.shape[0]: break
        if c==0:
            LossCats=-torch.mean(GTcats[c] * torch.log((Predclasslist[c] + 0.0000001)))
        else:
            LossCats+=-torch.mean(GTcats[c] * torch.log((Predclasslist[c] + 0.0000001)))

#.............................combined loss..............................................................................................................

    Loss=LossIOU*IOUConst+LossSeg+LossCats*CATConst

    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
#----------------Average loss--------------------------------------------------------------------------------------------------------------------------------------
    if AVGLossSeg==-1:  AVGLossSeg=float(LossSeg.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLossSeg=AVGLossSeg*0.999+0.001*float(LossSeg.data.cpu().numpy()) # Intiate runing average loss

    if AVGLossIOU==-1:  AVGLossIOU=float(LossIOU.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLossIOU=AVGLossIOU*0.999+0.001*float(LossIOU.data.cpu().numpy()) # Intiate runing average loss
    
    if AVGLossCat==-1:  AVGLossCat=float(LossCats.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLossCat=AVGLossCat*0.999+0.001*float(LossCats.data.cpu().numpy()) # Intiate runing average loss
    
    IOUConst=0.2*AVGLossSeg/AVGLossIOU

    CATConst =4*AVGLossSeg / AVGLossCat
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 2000 == 0:# and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/Learning_Rate_Init.npy",Learning_Rate_Init)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 10000 == 0 and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss

        txt="\n"+str(itr)+"\t Seg Loss "+str(AVGLossSeg)+"\t IOU Loss "+str(AVGLossIOU)+"\t"+"\t Cat Loss "+str(AVGLossCat)+"\t"+str(Learning_Rate)+" Init_LR"+str(Learning_Rate_Init)
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr%10000==0 and itr>=StartLRDecayAfterSteps:
        Learning_Rate-= Learning_Rate_Decay
        if Learning_Rate<=1e-6:
            Learning_Rate_Init-=1e-6
            if Learning_Rate_Init <= 1e-6: Learning_Rate_Init = 2e-6
            Learning_Rate=Learning_Rate_Init
            Learning_Rate_Decay=Learning_Rate/20
        print("Learning Rate="+str(Learning_Rate)+"   Learning_Rate_Init="+str(Learning_Rate_Init))
        print("======================================================================================================================")
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer
        torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
#----------------------------------------Evaluate-------------------------------------------------------------------------------------------
    if itr % 10000 == 0:
        print("Evaluating")
        Eval.Eval(Net,itr)

