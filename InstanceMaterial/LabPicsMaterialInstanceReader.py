## Reader foe material instances from the LabPics Dataset


import cv2
import numpy as np
import random
#############################################################################################
def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
##############################################################################################
CatName={}
CatName[1]='Vessel'
CatName[2]='V Label'
CatName[3]='V Cork'
CatName[4]='V Parts GENERAL'
CatName[5]='Ignore'
CatName[6]='Liquid GENERAL'
CatName[7]='Liquid Suspension'
CatName[8]='Foam'
CatName[9]='Gel'
CatName[10]='Solid GENERAL'
CatName[11]='Granular'
CatName[12]='Powder'
CatName[13]='Solid Bulk'
CatName[14]='Vapor'
CatName[15]='Other Material'
CatName[16]='Filled'
###############################################################################################


#Reader for the coco panoptic data set for pointer based image segmentation
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
import random
############################################################################################################
###########################Display image##################################################################
def show(Im,Name="img"):
    cv2.imshow(Name,Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
##############################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r"\ChemLabScapeDataset_Finished\Annotations\\", MaxBatchSize=100,MinSize=250,MaxSize=1000,MaxPixels=800*800*5,TrainingMode=True, ReadEmpty=True):

        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.ClassBalance=False  # Load each class with equal probability
# ----------------------------------------Create list of annotations arranged by class--------------------------------------------------------------------------------------------------------------
        self.AnnList = [] # Image/annotation list
        self.AnnByCat = {} # Image/annotation list by class

        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(MainDir):
           if ReadEmpty: SubDirs=["Material","EmptyRegions"] # Consider consider empty vessel regions as instances
           else: SubDirs=["Material"]
           for sdir in SubDirs:
                InstDir=MainDir+"/"+AnnDir+r"//"+sdir+"//"
                if not os.path.isdir(InstDir): continue
     #------------------------------------------------------------------------------------------------
                for Name in os.listdir(InstDir):
                    CatString=""
                    if "CatID_"in Name:
                           CatString=Name[Name.find("CatID_")+6:Name.find(".png")]
                    ListCat = []
                    if sdir=="EmptyRegions": ListCat=[0]
                    CatDic={}
                    CatDic["Image"]=MainDir+"/"+AnnDir+"/Image.png"
                    while (len(CatString)>0):
                        if "_" in CatString:
                            ID=int(CatString[:CatString.find("_")])
                        else:
                            ID=int(CatString)
                            CatString=""
                        if not ID in ListCat: ListCat.append(ID)
                        CatString=CatString[CatString.find("_")+1:]
                    CatDic["Cats"]=ListCat
                    CatDic["Ann"]=InstDir+"/"+Name
                    print(CatDic)
                    self.AnnList.append(CatDic)
                    for i in ListCat:
                        if i not in self.AnnByCat:
                            self.AnnByCat[i]=[]
                        self.AnnByCat[i].append(CatDic)
#------------------------------------------------------------------------------------------------------------
        if TrainingMode:
            for i in self.AnnByCat: # suffle
                    np.random.shuffle(self.AnnByCat[i])
            np.random.shuffle(self.AnnList)

        self.CatNum={}
        for i in self.AnnByCat:
              print(str(i)+") Num Examples="+str(len(self.AnnByCat[i])))
              self.CatNum[i]=len(self.AnnByCat[i])
        print("Total=" + str(len(self.AnnList)))
        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize(self,Img, AnnMap,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================

        h,w,d=Img.shape
        Bs = np.min((h/Hb,w/Wb))
        if Bs<1 or Bs>1.5:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(h / Bs)+1
            w = int(w / Bs)+1
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            AnnMap = cv2.resize(AnnMap, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
 # =======================Crop image to fit batch size===================================================================================


        if np.random.rand()<0.6:
            if w>Wb:
                X0 = np.random.randint(w-Wb)
            else:
                X0 = 0
            if h>Hb:
                Y0 = np.random.randint(h-Hb)
            else:
                Y0 = 0

            Img=Img[Y0:Y0+Hb,X0:X0+Wb,:]
            AnnMap = AnnMap[Y0:Y0+Hb,X0:X0+Wb,:]

        if not (Img.shape[0]==Hb and Img.shape[1]==Wb):
            Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
            AnnMap = cv2.resize(AnnMap, dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,AnnMap
        # misc.imshow(Img)
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize2(self,Img, Mask,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        Mk=(Mask[:, :, 0]>0)*(Mask[:, :, 0]<3).astype(np.uint8)
        bbox= cv2.boundingRect(Mk)
        [h, w, d] = Img.shape
        Rs = np.max((Hb / h, Wb / w))
        Wbox = int(np.floor(bbox[2]))  # Segment Bounding box width
        Hbox = int(np.floor(bbox[3]))  # Segment Bounding box height
        if Wbox==0: Wbox+=1
        if Hbox == 0: Hbox += 1


        Bs = np.min((Hb / Hbox, Wb / Wbox))
        if Rs > 1 or Bs<1 or np.random.rand()<0.3:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float)).astype(np.int64)

 # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox)-1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb)+1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox)-1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb)+1))

        if Ymax<=Ymin: y0=Ymin
        else: y0 = np.random.randint(low=Ymin, high=Ymax + 1)

        if Xmax<=Xmin: x0=Xmin
        else: x0 = np.random.randint(low=Xmin, high=Xmax + 1)

        # Img[:,:,1]*=Mask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        Mask = Mask[y0:y0 + Hb, x0:x0 + Wb]
#------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)
        if not (Mask.shape[0] == Hb and Mask.shape[1] == Wb):Mask = cv2.resize(Mask.astype(float), dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,Mask
        # misc.imshow(Img)
#################################################Generate Annotaton mask###################################################################
######################################################Augmented Image##################################################################################################################################
    def Augment(self,Img,AnnMap,prob):
        Img=Img.astype(np.float)
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            AnnMap = np.fliplr(AnnMap)
        if np.random.rand()<0.2: # flip left up down
            Img=np.flipud(Img)
            AnnMap = np.flipud(AnnMap)
        if np.random.rand() < 0.2:
            Img = np.rot90(Img)
            AnnMap = np.rot90(AnnMap)

        if np.random.rand()<0.5:
            Img = Img[..., :: -1]




        if np.random.rand() < prob: # resize
            r=r2=(0.3 + np.random.rand() * 1.7)
            if np.random.rand() < prob*2:
                r2=(0.5 + np.random.rand())
            h = int(Img.shape[0] * r)
            w = int(Img.shape[1] * r2)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            AnnMap =  cv2.resize(AnnMap, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        # if np.random.rand() < prob/3: # Add noise
        #     noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9
        #     Img *=noise
        #     Img[Img>255]=255
        #
        # if np.random.rand() < prob/3: # Gaussian blur
        #     Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < prob*2:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img>255]=255

        if np.random.rand() < prob:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,AnnMap
##################################################################################################################################################################
#Split binary mask correspond to a singele segment into connected components
    def GetConnectedSegment(self, Seg):
            [NumCCmp, CCmpMask, CCompBB, CCmpCntr] = cv2.connectedComponentsWithStats(Seg.astype(np.uint8))  # apply connected component
            Mask=np.zeros([NumCCmp,Seg.shape[0],Seg.shape[1]],dtype=bool)
            BBox=np.zeros([NumCCmp,4])
            Sz=np.zeros([NumCCmp],np.uint32)
            for i in range(1,NumCCmp):
                Mask[i-1] = (CCmpMask == i)
                BBox[i-1] = CCompBB[i][:4]
                Sz[i-1] = CCompBB[i][4] #segment Size
            return Mask,BBox,Sz,NumCCmp-1
############################################################################################################################
#################################################Generate Pointer mask#############################################################################################################333
    def GeneratePointermask(self, Mask):
        bbox = cv2.boundingRect(Mask.astype(np.uint8))
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        xmax = np.min([x1 + Wbox+1, Mask.shape[1]])
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height
        ymax = np.min([y1 + Hbox+1, Mask.shape[0]])
        PointerMask=np.zeros(Mask.shape,dtype=np.float)
        if Mask.max()==0:return PointerMask

        while(True):
            x =np.random.randint(x1,xmax)
            y = np.random.randint(y1, ymax)
            if Mask[y,x]>0:
                PointerMask[y,x]=1
                return(PointerMask)
########################################################################################################################################################
# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, pos, Hb=-1, Wb=-1):
# -----------------------------------Image and resize-----------------------------------------------------------------------------------------------------
            if self.ClassBalance: # pick with equal class probability
                while (True):
                     CL=random.choice(list(self.AnnByCat.keys()))
                     CatSize=len(self.AnnByCat[CL])
                     if CatSize>0: break

                Nim = np.random.randint(CatSize)
               # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
                Ann=self.AnnByCat[CL][Nim]
          #      print(Ann["Image"])

            else: # Pick with equal class probabiliry
                Nim = np.random.randint(len(self.AnnList))
                Ann=self.AnnList[Nim]
                CatSize=len(self.AnnList)
            #print(Ann)
            Img = cv2.imread(Ann["Image"])  # Load Image
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
            AnnMask=cv2.imread(Ann["Ann"])
            Cats=Ann["Cats"]


#******************************************************************************
#-------------------------Read annotation-------------------------------------------------------------------------------
#-------------------------Augment-----------------------------------------------------------------------------------------------
           # Img,AnnMask=self.Augment(Img,AnnMask,np.min([float(1000/CatSize)*0.5+0.06+1,1]))
#-----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            if not Hb==-1:
               Img, AnnMask = self.CropResize2(Img, AnnMask, Hb, Wb)
            if AnnMask.sum()<900:
                  self.LoadNext(pos, Hb, Wb)
                  return 1
#----------------------Generate forward and background segment mask-----------------------------------------------------------------------------------------------------------
  
 #***************************Split segment to  connected componenet*************************************************
            BG=AnnMask[:, :, 0] > 2
            FR=((AnnMask[:,:,0] > 0) * (AnnMask[:,:,0]  < 3)).astype(np.uint8)
            Mask,BBox,Sz,NumCCmp = self.GetConnectedSegment(FR)
            if NumCCmp>1:
                Ind=[]
                for i in range(NumCCmp):
                    if (Mask[i]>0).sum()>2000:
                        Ind.append(i)
                    else:
                        BG[Mask[i]]=1
                if len(Ind)>0:
                     BG[FR] = 1
                     FR=Mask[Ind[np.random.randint(len(Ind))]]
                     BG[FR] = 0
#----------------------Generate forward and background segment mask-----------------------------------------------------------------------------------------------------------

            self.BInstFR[pos] = FR #(AnnMask[:,:,0] > 0) * (AnnMask[:,:,0]  < 3)
            self.BInstBG[pos] = BG#AnnMask[:,:,0] > 2
#------------------------Generate ROI Mask------------------------------------------------------------------------------------------------------------------------------------
            if np.random.rand()<0.5:
                self.BROI[pos]=np.ones(self.BInstBG[pos].shape)
            else:
                if np.random.rand() < 0.5:
                    self.BROI[pos] = (self.BInstFR[pos] + (AnnMask[:, :, 1] > 0) * (AnnMask[:, :, 1] < 3) + (AnnMask[:, :, 2] > 0) * (AnnMask[:, :, 2] < 3))>0
                else:
                    self.BROI[pos] = (self.BInstFR[pos] + self.BInstBG[pos] + (AnnMask[:, :, 1] > 0) + (AnnMask[:, :, 2] > 0))>0
#-----------------------Generate Ignore mask-------------------------------------------------------------------------------------------------------
            self.BIgnore[pos] = (AnnMask[:, :, 2] == 7)

            self.BImg[pos] = Img
#-------------------------Generate Pointer mask-----------------------------------------------------------------------------------
            self.BPointerMask[pos] =  self.GeneratePointermask(self.BInstFR[pos])
#------------------------Category list------------------------------------------------------------------------------------------
            for c in Ann["Cats"]:
                    self.BCatList[pos][c]=1


############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        
        #====================Start reading data multithreaded===========================================================

        self.BIgnore = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BImg = np.zeros([BatchSize, Hb, Wb,3], dtype=float)
        self.BInstFR = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BInstBG = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BROI = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BPointerMask = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BCatList=np.zeros([BatchSize,len(CatName)], dtype=float)
        self.thread_list = []
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="thread"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
        self.itr+=BatchSize
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
    def LoadBatch(self):
# Load batch for training (muti threaded  run in parallel with the training proccess)
# return previously  loaded batch and start loading new batch
            self.WaitLoadBatch()
            Imgs=self.BImg
            Ignore=self.BIgnore
            InstFR=self.BInstFR
            InstBG=self.BInstBG
            ROI=self.BROI
            PointerMask=self.BPointerMask
            CatList=self.BCatList
            self.StartLoadBatch()
            return Imgs, Ignore, InstFR, InstBG, ROI,PointerMask,CatList

############################Load single data with no augmentation############################################################################################################################################################
    def LoadSingle(self):
       # print(self.itr)
        if self.itr>=len(self.AnnList):
            self.epoch+=1
            self.itr=0
        Ann = self.AnnList[self.itr]
        self.itr+=1
       #.................Load Files ....................................................................
        Img = cv2.imread(Ann["Image"])  # Load Image
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
        AnnMask = cv2.imread(Ann["Ann"])
       #...............Procesess Files...................................................................
       # ***************************Split segment to  connected componenet*************************************************

        BG = AnnMask[:, :, 0] > 2
        FR = ((AnnMask[:, :, 0] > 0) * (AnnMask[:, :, 0] < 3)).astype(np.uint8)
        Mask, BBox, Sz, NumCCmp = self.GetConnectedSegment(FR)
        if NumCCmp > 1:
            Ind = []
            for i in range(NumCCmp):
                if (Mask[i]>0).sum()>2000:
                    Ind.append(i)
                else:
                    BG[Mask[i]] = 1
            if len(Ind) > 0:
                BG[FR] = 1
                FR = Mask[0]
                BG[FR] = 0

        # ------------------------Generate ROI Mask------------------------------------------------------------------------------------------------------------------------------------
        ROI = np.ones(FR.shape)
        # -----------------------Generate Ignore mask-------------------------------------------------------------------------------------------------------
        Ignore = (AnnMask[:, :, 2] == 7)


        PointerPoint=self.GeneratePointermask(FR)


        CatList = np.zeros([len(CatName)], dtype=float)
        for c in Ann["Cats"]:
            CatList[c] = 1
        return Img, FR ,BG,ROI,PointerPoint,Ignore,CatList, self.itr>=len(self.AnnList)
