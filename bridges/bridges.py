#!/usr/bin/python
#David Fouhey
#Bridges of Allegheny County with Caffe64
import numpy as np
import os

localFile = "bridges.data.version2"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bridges/bridges.data.version2"
targetName = ["Through/Deck","Material","Span","Rel-L","Type"]
xinds, yinds = range(1,8), range(8,13)
numRuns = 100

if __name__ == "__main__":
    print "Bridges of Allegheny County"

    if not os.path.exists(localFile):
        print "Downloading data"
        os.system("wget %s" % url)

    #read it
    lines = [l.split(",") for l in file(localFile).read().strip().split("\n")]

    N = len(lines)
    teFrac = int(N*0.1)

    #Get X
    feats = []
    for j in xinds:
        #one-hot the Xs
        ex = [lines[i][j] for i in range(N)]
        exValues = list(set(ex))
        exValues.sort()

        for v in exValues:
            Y = np.expand_dims(np.array([1 if ex[i] == v else 0 for i in range(N)]),axis=1)
            feats.append(Y)
    Xs = np.concatenate(feats,axis=1)

    #Get Y, where 0 is unknown
    Ys = []
    for j in yinds:
        ex = [lines[i][j] for i in range(N)]
        exValues = list(set(ex))
        exValues.sort()
        assert exValues[0] == "?"
        Y = np.zeros((N,1))
        for i in range(N):
            for j in range(len(exValues)):
                if ex[i] == exValues[j]:
                    Y[i] = j
        Ys.append(Y)        
    Ys = np.concatenate(Ys,axis=1)

    #for each y, write out a network
    for j in range(Ys.shape[1]):
        y = Ys[:,j]
        numClass = int(np.max(y))

        base = file("bridgenet_base.txt").read()
        net = file("bridgenet.txt","w").write(base % numClass)

        res = []
        for i in range(numRuns):
            p = np.random.permutation(N)
            te, tr = p[:teFrac], p[teFrac:]

            Xr, Xe = Xs[tr,:], Xs[te,:]
            yr, ye = y[tr], y[te]

            #get rid of unlabeled values
            kr, ke = yr!= 0, ye != 0
            Xr,Xe = Xr[kr,:], Xe[ke,:]
            yr,ye = yr[kr]-1, ye[ke]-1

            np.savetxt('Xr.txt',Xr,'%f',delimiter=" ")
            np.savetxt('yr.txt',yr,'%f',delimiter=" ")
            np.savetxt('Xe.txt',Xe,'%f',delimiter=" ")
            np.savetxt('ye.txt',ye,'%f',delimiter=" ")

            com = "../caffe64 train bridgenet.txt bridge.bin bridge_optimsetting.txt %d Xr.txt yr.txt > /dev/null" % yr.shape[0]
            os.system(com)
            com = "../caffe64 test bridgenet.txt bridge.bin %d Xe.txt yep.txt > /dev/null " % ye.shape[0]
            os.system(com)

            yh = np.argmax(np.loadtxt("yep.txt"),axis=1)
            res.append(np.mean(ye==yh))

        res = np.array(res)
        print "Average Accuracy for target %s: %f" % (targetName[j], np.mean(res))
        toClean = ["Xr.txt","yr.txt","Xe.txt","ye.txt","yep.txt","bridgenet.txt","bridge.bin"]
        map(os.remove,toClean)


