#!/usr/bin/python
#David Fouhey
#Credit to Andrej Karpathy for the idea
#https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html

import numpy as np
import cv2, pdb, os, sys

#Two network options are provided
NETWORK_NAME = "networkReg.txt"
OPTIMSETTING_NAME = "optimsetting.txt"

def setup(imName,expName=""):
    I = cv2.imread(imName).astype(np.float)
    I = (I-128)/128
    X,Y = np.meshgrid(np.linspace(-1,1,I.shape[1]),np.linspace(-1,1,I.shape[0]))
    h, w  = I.shape[0], I.shape[1]
    R,G,B = I[:,:,0], I[:,:,1], I[:,:,2]

    C = np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],axis=1)

    np.savetxt(expName+"im_X.txt",C,fmt="%f",delimiter=" ",)
    np.savetxt(expName+"im_Yr.txt",R.ravel(),fmt="%f",delimiter=" ")
    np.savetxt(expName+"im_Yg.txt",G.ravel(),fmt="%f",delimiter=" ")
    np.savetxt(expName+"im_Yb.txt",B.ravel(),fmt="%f",delimiter=" ")

    sz = h*w

    trc = "../caffe64 train %s %sim_%%s.bin %s %d %sim_X.txt %sim_Y%%s.txt" % (NETWORK_NAME,expName,OPTIMSETTING_NAME,sz,expName,expName)
    tec = "../caffe64 test %s %sim_%%s.bin %d %sim_X.txt %sim_Y%%sr.txt" % (NETWORK_NAME,expName,sz,expName,expName)

    comf = file(expName+"com.sh","w")
    for c in "rgb":
        comf.write(trc % (c,c)+"\n")
        comf.write(tec % (c,c)+"\n")

def run(expName=""):
    os.system("bash %scom.sh" % expName)


def reconstructTo(imName,expName=""):
    I = cv2.imread(imName)
    h, w = I.shape[0], I.shape[1]

    reco = lambda x: np.expand_dims(x.reshape(h,w),axis=2)

    Yr = reco(np.loadtxt(expName+"im_Yrr.txt"))
    Yg = reco(np.loadtxt(expName+"im_Ygr.txt"))
    Yb = reco(np.loadtxt(expName+"im_Ybr.txt"))
    Ir = np.clip(np.concatenate([Yr,Yg,Yb],axis=2),-1,1)
    Ir = (Ir*128)+128
    cv2.imwrite(imName+"_reco.png",Ir)

def cleanup(expName=""):
    toRm = ["im_X.txt","com.sh"]
    for c in "rgb":
        toRm += ["im_Y%s.txt" % c, "im_Y%sr.txt" % c,"im_%s.bin" % c]
    toRm = [expName+f for f in toRm]
    for f in toRm:
        try:
            os.remove(f)
        except:
            print "Can't delete",f

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "%s image expName" % sys.argv[0]
        sys.exit(1)
    im, exp = sys.argv[1], sys.argv[2]
    setup(im,exp)
    run(exp)
    reconstructTo(im,exp)
    cleanup(exp)
