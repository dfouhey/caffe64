#!/usr/bin/python
#David Fouhey
#Basic Markovian text generator
#
#Some possible samples:
#   1) Moby Dick 
#   2) The intel software manuals (convert to txt with poppler)
#   3) ../caffe64.s
#   4) Paper titles, from Arxiv (see titles)
#   5) The Aeneid
#
#We provide pretrained models for a window of 10 in models/
#
#   1) moby
#   2) intel
#   3) c64
#   4) title
#   5) aeneid

import numpy as np
import sys, os

#how many characters as a window
WINDOW_SIZE = 10
#how big
NUM_OUTPUTS = 50
REMOVENL = False

#where to store temp files
TEMP_PATH = "./"

#template for network
NET_NAME = "charnet_base.txt"
#optimization settings
OPTIM_NAME = "optimsetting.txt"

force = map(chr,range(ord('a'),ord('z')+1))

class SymbolLUT:
    def __init__(self):
        self.lut, self.lutL = [], 0

    def encode(self,c):
        #ok -1 seems weird but lutL is the length of the symbol output
        #so it's one more than the number of actual symbols
        return self.lut.index(c) if c in self.lut else self.lutL-1

    def decode(self,i):
        return self.lut[i] if i < len(self.lut) else '~'

    def build(self,count,S,force=[]):
        symb = list(set(S))
        symb.sort(key = lambda c:-S.count(c))
        extras = [s for s in symb if s not in force]
        self.lut = force+extras 
        if len(self.lut) > count:
            self.lut = self.lut[:count-1]
        self.lutL = len(self.lut)+1

    def load(self,lut):
        self.lut, self.lutL = list(lut), len(lut)+1

    def serialize(self):
        return ''.join(self.lut)

def vectorW(w,LUT):
    nc = len(w)
    f = LUT.lutL
    v = np.zeros((1,nc*f),np.uint8)
    for i in range(nc):
        v[0,i*f+LUT.encode(w[i])] = 1
    return v  

def die():
    print "charmodel train textname numSamples modelname"
    print "    or"
    print "charmodel sample modelname numSamples temperature"
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        die()

    if sys.argv[1] == "train":
        #charmodel train textname numSamples modelname 
        fileName = sys.argv[2]
        numSamples = int(sys.argv[3])
        modelName = sys.argv[4]
        modelNameUse = modelName.replace("/","_")

        binName = modelName+".bin"
        lutName = modelName+".txt"

        S = file(fileName).read()
        S = S.decode('utf8').encode('ascii','replace').lower().replace("\t"," ").replace("\r","")
        if REMOVENL:
            S = S.replace("\n","")

        while S.find("  ") != -1:
            S = S.replace("  "," ")

        LUT = SymbolLUT()
        LUT.build(NUM_OUTPUTS,S,force=force)

        Xs = []
        y = np.zeros((numSamples,1))

        for si in range(numSamples):
            if si % 10000 == 0:
                print "Generating %8d / %8d" % (si,numSamples)
            selectStart = WINDOW_SIZE+1
            loc = np.random.choice(len(S)-selectStart)+selectStart
            v = S[loc]
            vx = S[loc-WINDOW_SIZE:loc]
            y[si] = LUT.encode(v)
            Xs.append(vectorW(vx,LUT))

        print "Concatenating"
        X = np.concatenate(Xs)

        print "Set up problem with %d instances and %d features" % (X.shape[0],X.shape[1])

        xf = TEMP_PATH+'/'+modelNameUse+'mbXr.txt'
        yf = TEMP_PATH+'/'+modelNameUse+'mbyr.txt'

        netf = TEMP_PATH+"/"+modelNameUse+"net.txt"
        template = file(NET_NAME).read()
        file(netf,"w").write(template % (WINDOW_SIZE*NUM_OUTPUTS,NUM_OUTPUTS))


        print "Writing"
        np.savetxt(xf,X,'%d',delimiter=' ')
        np.savetxt(yf,y,'%d',delimiter=' ')
        file(lutName,"w").write(LUT.serialize())

        
        os.system("../caffe64 train %s %s %s %d %s %s" % (netf,binName,OPTIM_NAME,X.shape[0],xf,yf))

        os.remove(xf)
        os.remove(yf)
        os.remove(netf)

    elif sys.argv[1] == "sample":
        #charmodel modelname temperature
        modelName = sys.argv[2]
        modelNameUse = modelName.replace("/","_")
        numSamples = int(sys.argv[3])
        temperature = float(sys.argv[4])


        print "Give me something to work with"
        print "v"*WINDOW_SIZE
        starterText = raw_input()

        binName = modelName+".bin"
        lutName = modelName+".txt"
        netf = TEMP_PATH+"/"+modelNameUse+"net.txt"

        template = file(NET_NAME).read()
        file(netf,"w").write(template % (WINDOW_SIZE*NUM_OUTPUTS,NUM_OUTPUTS))

        LUT = SymbolLUT()
        LUT.load(file(lutName).read())

        Xs = []

        cs = starterText[:WINDOW_SIZE]

        print "Sampling"
        res = cs
        for i in range(numSamples):
            if i % 10 == 0:
                print i
            x = vectorW(cs,LUT)

            np.savetxt(modelNameUse+'xt.txt',x,'%d',delimiter=' ')
            os.system("../caffe64 test %s %s 1 %sxt.txt %syep.txt >/dev/null" % (netf,binName,modelNameUse,modelNameUse))
            yp = np.loadtxt(modelNameUse+"yep.txt")
            yp = yp**temperature
            yp = yp / np.sum(yp)
            i = np.random.choice(len(yp),p=yp)
            yh = LUT.decode(i)
            cs = cs[1:]+yh
            res += yh
        print res
        os.remove(modelNameUse+"xt.txt")
        os.remove(modelNameUse+"yep.txt")
        os.remove(netf)


