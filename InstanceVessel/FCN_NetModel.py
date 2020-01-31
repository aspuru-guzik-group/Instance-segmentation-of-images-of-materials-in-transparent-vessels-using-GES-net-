# Fully convolutional neural net GES net /pointer net, given an image a point in the imagee and an ROI mask, the net predict the region/segment of the object/stuff containing the pointer th score of the segment predicition (IOU/Accuracy), and wether this segment is vessel or other object/image (only in case were the net is trained with COCO)
import scipy.misc as misc
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
######################################################################################################################333
class Net(nn.Module):
#########################################Build net###############################################################################
    def __init__(self, NumClasses=2):
        # Generate standart FCN net for image segmentation with only image as input (attention layers will be added at next function
        # --------------Build layers for standart FCN with only image as input------------------------------------------------------
            super(Net, self).__init__()
            # ---------------Load pretrained  Resnet encoder----------------------------------------------------------
            self.Encoder = models.resnet101(pretrained=True)
            # ---------------Create Pyramid Scene Parsing PSP layer -------------------------------------------------------------------------
            self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]

            self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder
            for Ps in self.PSPScales:
                self.PSPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 1024, stride=1, kernel_size=3, padding=1, bias=True)))
                # nn.BatchNorm2d(1024)))
            self.PSPSqueeze = nn.Sequential(
                nn.Conv2d(4096, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()))
            # ------------------Skip squeeze applied to the (concat of upsample+skip conncection layers)-----------------------------------------------------------------------------
            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 128, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()))


            # ----------------Build ROI and Pointer procsscing layers------------------------------------------------------------------------------------------
            self.FinalPrdiction = nn.Conv2d(128, NumClasses, stride=1, kernel_size=3, padding=1, bias=False)

            self.AttentionLayers = nn.ModuleList()
            self.ROIEncoder = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True)
            self.ROIEncoder.bias.data = torch.zeros(self.ROIEncoder.bias.data.shape)
            self.ROIEncoder.weight.data = torch.zeros(self.ROIEncoder.weight.data.shape)

            self.PointerEncoder = nn.Conv2d(1, 64, stride=1, kernel_size=3, padding=1, bias=True)
            self.PointerEncoder.bias.data = torch.zeros(self.ROIEncoder.bias.data.shape)
            self.PointerEncoder.weight.data = torch.ones(self.ROIEncoder.weight.data.shape)
##########################################Add final classificatior and evaluator layers#############################################################################################################
    def AddEvaluationClassificationLayers(self,NumClass=1):
       self.ClassLayer1 = nn.Sequential(nn.Conv2d(2048, 1024,stride=2, kernel_size=3, padding=1, bias=True),nn.ReLU())
       self.ClassLayer2 = nn.Sequential(nn.Conv2d(1024, 512,stride=2, kernel_size=3, padding=1, bias=True),nn.ReLU(),nn.AdaptiveAvgPool2d(output_size=(1, 1)))
       self.ClassLayer3 = nn.Sequential(nn.Linear(in_features=512, out_features=NumClass*2, bias=False))

       self.EvalLayer1 = nn.Sequential(nn.Conv2d(2048, 1024,stride=2, kernel_size=3, padding=1, bias=True),nn.ReLU())
       self.EvalLayer2 = nn.Sequential(nn.Conv2d(1024, 512,stride=2, kernel_size=3, padding=1, bias=True),nn.ReLU(),nn.AdaptiveAvgPool2d(output_size=(1, 1)))
       self.EvalLayer3 = nn.Sequential(nn.Linear(in_features=512, out_features=1, bias=False))

    #   self.cuda()
##########################################################################################################################################################
    def forward(self,Images,Pointer,ROI,UseGPU=True,TrainMode=True,FreezeBatchNorm_EvalON=False):

#----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68,116.779,103.939]
                RGBStd = [65,65,65]
                if TrainMode==True:
                    tp=torch.FloatTensor
                else:
                    tp=torch.half
              #      self.eval()
                    self.half()
                if FreezeBatchNorm_EvalON: self.eval()


                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2,3).transpose(1, 2).type(tp)

#-------------------Convert ROI mask and pointer point mask into pytorch format----------------------------------------------------------------
                ROImap = torch.autograd.Variable(torch.from_numpy(ROI.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(tp)
                Pointermap = torch.autograd.Variable(torch.from_numpy(Pointer.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(tp)
#---------------Convert to cuda gpu-------------------------------------------------------------------------------------------------------------------
                if UseGPU:
                    ROImap=ROImap.cuda()
                    Pointermap=Pointermap.cuda()
                    InpImages=InpImages.cuda()
                    self.cuda()
                else:
                    ROImap = ROImap.cpu().float()
                    Pointermap = Pointermap.cpu().float()
                    InpImages = InpImages.cpu().float()
                    self.cpu().float()

#----------------Normalize image values-----------------------------------------------------------------------------------------------------------
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
#---------------Run Encoder first layer-----------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
#------------------------Convery ROI mask and pointer map into attention layer and merge with image feature mask-----------------------------------------------------------
                r = self.ROIEncoder(ROImap) # Generate attention map from ROI mask
                pt = self.PointerEncoder(Pointermap) # Generate attention Mask from Pointer point
                sp = (x.shape[2], x.shape[3])
                pt = nn.functional.interpolate(pt, size=sp, mode='bilinear')  #
                r = nn.functional.interpolate(r, size=sp, mode='bilinear')  # Resize
                x = x* pt + r # Merge feature mask and attention maps
#-------------------------Run remaining encoder layer------------------------------------------------------------------------------------------
                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                EncoderMap = self.Encoder.layer4(x)
#------------------Run psp  Layers----------------------------------------------------------------------------------------------
                PSPSize=(EncoderMap.shape[2],EncoderMap.shape[3]) # Size of the original features map

                PSPFeatures=[] # Results of various of scaled procceessing
                for i,PSPLayer in enumerate(self.PSPLayers): # run PSP layers scale features map to various of sizes apply convolution and concat the results
                      NewSize=(np.array(PSPSize)*self.PSPScales[i]).astype(np.int)
                      if NewSize[0] < 1: NewSize[0] = 1
                      if NewSize[1] < 1: NewSize[1] = 1

                      # print(str(i)+")"+str(NewSize))
                      y = nn.functional.interpolate(EncoderMap, tuple(NewSize), mode='bilinear')
                      #print(y.shape)
                      y = PSPLayer(y)
                      y = nn.functional.interpolate(y, PSPSize, mode='bilinear')

                #      if np.min(PSPSize*self.ScaleRates[i])<0.4: y*=0
                      PSPFeatures.append(y)
                x=torch.cat(PSPFeatures,dim=1)
                x=self.PSPSqueeze(x)
#----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear') #Resize
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1)
                  x = self.SqueezeUpsample[i](x)
#---------------------------------Final prediction segment mask-------------------------------------------------------------------------------
                x = self.FinalPrdiction(x) # Make prediction per pixel
                x = nn.functional.interpolate(x,size=InpImages.shape[2:4],mode='bilinear') # Resize to original image size
#********************************************************************************************************
                #x = nn.UpsamplingBilinear2d(size=InpImages.shape[2:4])(x)
                Prob=F.softmax(x,dim=1) # Calculate class probability per pixel
                tt,Labels=x.max(1) # Find label per pixel
#------------------------------Evaluaion layer predict net IOU score -----------------------------------------------------------------------------------------------
                x=self.EvalLayer1(EncoderMap)
                x=self.EvalLayer2(x)[:,:,0,0]
                IOU=self.EvalLayer3(x)
#------------------------------Is the  segment a vesssel or not (only used when the net is trained with COCO as additional dataset)-----------------------------------------------------------------------------------------------------------
                x=self.ClassLayer1(EncoderMap)
                x=self.ClassLayer2(x)[:,:,0,0]
                x=self.ClassLayer3(x)
                IsVessel=F.softmax(x,dim=1)
                return Prob,Labels,IOU, IsVessel # IsVessel not used in this case







