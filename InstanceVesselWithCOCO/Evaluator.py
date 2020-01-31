# Evaluate net accuracy on test set


import numpy as np
import os
import LabPicsVesselInstanceReader as ChemScapeInstanceReader
import torch

############################################################################################################
#########################################################################################################################
class Evaluator:
    def __init__(self, AnnDir,OutFile): # Make reader for test set, and open file for eevaluation results
        self.AnnDir = AnnDir
        self.OutFile=OutFile
        if not os.path.exists(OutFile):
            f=open(OutFile,"w")
            f.close()
        print("-------------------------------------Creating test evaluator------------------------------------------------------")
        self.Reader = ChemScapeInstanceReader.Reader(MainDir=self.AnnDir, TrainingMode=False)
#####################################################################################333
    def Eval(self,Net,itr):
        print("Evaluating")
        Finished=False

        IOUSum = 0
        InterSum = 0
        UnionSum = 0
        ImSum=0
        IOUDif = 0
        CatAccuracy = 0

        # IOUSumCat = np.zeros([20])
        # InterSumCat = np.zeros([20])
        # UnionSumCat = np.zeros([20])
        # ImSumCat = np.zeros([20])
        #

        while (not Finished):
                Imgs, AnnMapGt, BG,ROI,PointerMap, Ignore, Cats, Finished=self.Reader.LoadSingle()
                # --------------------------------------
                # Imgs[:, :, 0] *= 1 - AnnMapGt.astype(np.uint8)
                # Imgs[:, :, 1] *= 1 - Ignore.astype(np.uint8)
                # print(Cats)
                # misc.imshow(Imgs)
                # misc.imshow((ROI + AnnMapGt * 2 + PointerMap * 3).astype(np.uint8) * 40)
                # print(ROI.shape)
                # ----------------------------------------------
                Imgs=np.expand_dims(Imgs,axis=0)
                PointerMap = np.expand_dims(PointerMap,axis=0)
                ROI = np.expand_dims(ROI, axis=0)
                with torch.autograd.no_grad():
                         Prob, LbPred, PredIOU, PredIsVessel  = Net.forward(Images=Imgs, Pointer=PointerMap,ROI=ROI)  # Run net inference and get prediction

                         PredIOU = np.squeeze(PredIOU.data.cpu().numpy())
                         Pred= LbPred.data.cpu().numpy()[0]*(1-Ignore)
                         GT=AnnMapGt*(1-Ignore)
                         Inter=(Pred*GT).sum()
                         Union=(Pred).sum()+(GT).sum()-Inter
                         if Union.sum()>0: #Union>0:
                            IOUSum += Inter/Union
                            InterSum += Inter
                            UnionSum += Union
                            ImSum += 1
                            IOUDif += np.abs(Inter / Union - PredIOU)
                            CatAccuracy += (PredIsVessel.data.cpu().numpy()>0).mean()
        #                     for k in Cats:
        #                         IOUSumCat[k] += Inter / Union
        #                         InterSumCat[k] += Inter
        #                         UnionSumCat[k]+= Union
        #                         ImSumCat[k] += 1
        #
        #                     # if GT.sum()>0:
        #                     #     print(k)
        #                     #     Im=Imgs[0].copy()
        #                     #     print( Inter / Union)
        #                     #     Im[:, :, 0] *= 1 - GT.astype(np.uint8)
        #                     #     Im[:, :, 2] *= (1-Ignore).astype(np.uint8)
        #                     #     Im[:, :, 1] *= 1 - Pred.astype(np.uint8)
        #                     #     misc.imshow(Im)
        #                  # break
        #
        #
        f = open(self.OutFile, "a")
        txt="\n====================="+str(itr)+"==============================================\n"
        # txt+=str(itr)+"\n"
        # PerPixelPerCat = []
        # PerImagePerCat = []
        # for nm in range(IOUSumCat.shape[0]):
        #     if UnionSumCat[nm]>0:
        #         txt += str(nm) + "\t" +CatName[nm]+"\t"
        #         txt += "IOU Average Per Pixel=\t"+str(InterSumCat[nm]/UnionSumCat[nm])+"\t"
        #         txt += "IOU Average Per Image=\t" + str(IOUSumCat[nm]/ImSumCat[nm])+"\tNum Examples\t"+str(ImSumCat[nm])+"\n"
        #         PerPixelPerCat.append(InterSumCat[nm]/UnionSumCat[nm])
        #         PerImagePerCat.append(IOUSumCat[nm]/ImSumCat[nm])


        txt += "\n\n Total IOU Average Per Pixel=\t" + str(InterSum / UnionSum) + "\t"
        txt += "Total IOU Average Per Image=\t" + str(IOUSum / ImSum) + "\n"
        txt +="\n------EVAL------\n"
        txt += "\n\n Dif Pred IOU=\t" + str(IOUDif / ImSum) + "\t"
        txt += "\n------Category------\n"
        txt += "Accuracy Rate Cat=\t" + str(CatAccuracy/ImSum) + "\n"
        # txt += "\n\n Cat Total IOU Average Per Pixel=\t" +  str(np.mean(PerPixelPerCat)) + "\t"
        # txt += "Cat Total IOU Average Per Image=\t" + str(np.mean(PerImagePerCat)) + "\n"
        f.write(txt)
        f.close()
        print(txt)






