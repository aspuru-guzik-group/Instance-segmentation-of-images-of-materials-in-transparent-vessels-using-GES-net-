## Evaluator for evaluating net accuracy against test data

import numpy as np
import os

import LabPicsMaterialInstanceReader as LabPicsInstanceReader
import torch
CatName={}
CatName[0]='Empty'
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
############################################################################################################
####################################Create datat Reader and statitics file#####################################################################################
class Evaluator:
    def __init__(self, AnnDir,OutFile):
        self.AnnDir = AnnDir
        self.OutFile=OutFile
        if not os.path.exists(OutFile):
            f=open(OutFile,"w")
            f.close()
        print("-------------------------------------Creating test evaluator------------------------------------------------------")
        self.Reader = LabPicsInstanceReader.Reader(MainDir=self.AnnDir, TrainingMode=False)
#################################################Evaluate net accuracy####################################333
    def Eval(self,Net,itr):
        print("Evaluating")
        Finished=False

        IOUSum = 0
        InterSum = 0
        UnionSum = 0
        ImSum=0
        IOUDif = 0
        CatTP = 0
        CatFP = 0
        CatFN = 0


        IOUSumCat = np.zeros([20])
        InterSumCat = np.zeros([20])
        UnionSumCat = np.zeros([20])
        ImSumCat = np.zeros([20])
        IOUDifCat = np.zeros([20])

        CatTPCat= np.zeros([20])
        CatFPCat= np.zeros([20])
        CatFNCat = np.zeros([20])

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
                         Prob, LbPred, PredIOU, Predclasslist = Net.forward(Images=Imgs, Pointer=PointerMap,ROI=ROI)  # Run net inference and get prediction

                         PredIOU=np.squeeze(PredIOU.data.cpu().numpy())
                         Pred= LbPred.data.cpu().numpy()[0]*(1-Ignore)
                         GT=AnnMapGt*(1-Ignore)
                         Inter=(Pred*GT).sum()
                         Union=(Pred).sum()+(GT).sum()-Inter
                         if Union>0:
                            IOUSum += Inter/Union
                            InterSum += Inter
                            UnionSum += Union
                            IOUDif+=np.abs(Inter/Union-PredIOU)
                            ImSum += 1
                            for k in range(len(Cats)):
                                if Cats[k]>0:
                                    k=int(k)
                                    IOUSumCat[k] += Inter / Union
                                    InterSumCat[k] += Inter
                                    UnionSumCat[k]+= Union
                                    ImSumCat[k] += 1
                                    IOUDifCat[k]+=np.abs(Inter/Union-PredIOU)
                                if Cats[int(k)]>0:
                                    if (Predclasslist[int(k)][0][1] > 0.5).data.cpu().numpy()>0:
                                        CatTPCat[k]+=1
                                        CatTP+=1
                                    else:
                                        CatFNCat[k] += 1
                                        CatFN += 1
                                else:
                                    if (Predclasslist[int(k)][0][1] > 0.5).data.cpu().numpy()>0:
                                        CatFPCat[k]+=1
                                        CatFP+=1


                            # if GT.sum()>0:
                            #     print(k)
                            #     Im=Imgs[0].copy()
                            #     print( Inter / Union)
                            #     Im[:, :, 0] *= 1 - GT.astype(np.uint8)
                            #     Im[:, :, 2] *= (1-Ignore).astype(np.uint8)
                            #     Im[:, :, 1] *= 1 - Pred.astype(np.uint8)
                            #     misc.imshow(Im)
                         # break


        f = open(self.OutFile, "a")
        txt="\n=================================================================================\n"
        txt+=str(itr)+"\n"
        PerPixelPerCat = []
        PerImagePerCat = []
        MeanDifIOUPerCat =[]
        for nm in range(IOUSumCat.shape[0]):
            if UnionSumCat[nm]>0:
                txt += str(nm) + "\t" +CatName[nm]+"\t"
                txt += "IOU Average Per Pixel=\t"+str(InterSumCat[nm]/UnionSumCat[nm])+"\t"
                txt += "IOU Average Per Image=\t" + str(IOUSumCat[nm]/ImSumCat[nm])+"\tNum Examples\t"+str(ImSumCat[nm])+"\t"
                txt += "IOU Eval Pred Error=\t" + str(IOUDifCat[nm]/ImSumCat[nm])+"\t"
                txt += "Accuracy Rate Cat=\t"+str(CatTPCat[nm]/(CatTPCat[nm]+CatFNCat[nm]+CatFPCat[nm]+0.0001)) +  "Recall Rate Cat=\t"+str(CatTPCat[nm]/(CatTPCat[nm]+CatFNCat[nm]+0.0001)) +  "Precision Rate Cat=\t"+str(CatTPCat[nm]/(CatTPCat[nm]+CatFPCat[nm]+0.0001)) + "\n"
                PerPixelPerCat.append(InterSumCat[nm]/UnionSumCat[nm])
                PerImagePerCat.append(IOUSumCat[nm]/ImSumCat[nm])
                MeanDifIOUPerCat.append(IOUDifCat[nm]/ImSumCat[nm])

        txt += "\n------Segmentation Accuracy------\n"
        txt += "\n\n Total IOU Average Per Pixel=\t" + str(InterSum / UnionSum) + "\t"
        txt += "Total IOU Average Per Image=\t" + str(IOUSum / ImSum) + "\n"

        txt += "\n\n Cat Total IOU Average Per Pixel=\t" +  str(np.mean(PerPixelPerCat)) + "\t"
        txt += "Cat Total IOU Average Per Image=\t" + str(np.mean(PerImagePerCat)) + "\n"
        txt +="\n------EVAL------\n"
        txt += "\n\n Dif Pred IOU=\t" + str(IOUDif / ImSum) + "\t"
        txt += "Dif Pred IOU Per Cat=\t" + str(np.mean(MeanDifIOUPerCat)) + "\n"
        txt += "\n------Category------\n"
        txt += "Accuracy Rate Cat=\t" + str(
            CatTP / (CatTP + CatFN + CatFP+0.0001)) + "\tRecall Rate Cat=\t" + str(
            CatTP / (CatTP + CatFN+0.0001)) + "\tPrecision Rate Cat=\t" + str(
            CatTP / (CatTP + CatFP+0.0001)) + "\n"



        f.write(txt)
        f.close()
        print(txt)






