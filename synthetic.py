import numpy
from sklearn.cluster import KMeans
import math


global M
M=18


def ProcessRawData():


    xTrain = numpy.genfromtxt('input_synth.csv', delimiter=',')
    xTrain = xTrain[:16000,:]
    
    yTrain = numpy.genfromtxt('output_synth.csv', delimiter=',')
    yTrain = GetProcessedY(yTrain[:16000])

    xTest = numpy.genfromtxt('input_synth.csv', delimiter=',')
    xTest = xTest[16000:18000,:]

    yTest = numpy.genfromtxt('output_synth.csv', delimiter=',')
    yTest = GetProcessedY(yTest[16000:18000])

    xVali = numpy.genfromtxt('input_synth.csv', delimiter=',')
    xVali = xVali[18000:,:]
    
    yVali = numpy.genfromtxt('output_synth.csv', delimiter=',')
    yVali = GetProcessedY(yVali[18000:])

    return xTrain,yTrain,xTest,yTest,xVali,yVali


def GetProcessedY(data):
    rows = data.shape[0]

    result = numpy.zeros(shape=(rows,1))
    for i in range(0,rows):
        result[i][0] = float(data[i])
    return result


def ClosedFormSolution(y,phi,regConst):
   
    phiT = numpy.transpose(phi)
    phiTphi = numpy.dot(phiT,phi)
    reg = numpy.identity(M)
    reg = numpy.multiply(regConst,reg)
    phiTphiInv = numpy.linalg.inv(phiTphi + reg)
    phiTphiInvPhi = numpy.dot(phiTphiInv,phiT)
    phiTphiInvPhiT = numpy.dot(phiTphiInvPhi,y)
    return phiTphiInvPhiT

def TestParams(w,mean,varInv,xTest,yTest):
    wtrans = numpy.transpose(w)
    pos = 0.0
    neg = 0.0
    rows,cols = yTest.shape
    for i in range(rows):
        phi = GetSingleBasisFunc(mean,varInv,xTest[i])
        train = numpy.dot(wtrans,phi)
        if(yTest[i] == round(numpy.sum(train))):
            pos+=1
        else:
            neg+=1
    regConst = 0
    accuracy = (pos/ (pos + neg)) * 100
    print ("For Lambda = %d , We have %d Accurate Predictions and %d Wrong Predictions, So Accuracy = %f" %(regConst,pos,neg,accuracy))
    return accuracy

def GetBasisFunction(x):
    rows,cols = x.shape
    
    kmeans = KMeans(n_clusters=M, random_state=0).fit(x)
    means =  kmeans.cluster_centers_
        
    variance = numpy.var(x,axis=0)
    varianceMatrix = numpy.zeros(shape=(10,10))
    for i in range(10):
        for j in range(10):
            if i==j:
                varianceMatrix[i][j] = variance[i]/10
                if(varianceMatrix[i][j] == 0) :
                    varianceMatrix[i][j] = 0.001
    varInv = numpy.linalg.inv(varianceMatrix)
    
    phi = numpy.zeros(shape=(rows,M))

    for i in range(rows):
        curX = x[i]
        for j in range(M):
            curMu = means[j]
            diff = numpy.subtract(curX,curMu)
            temp = numpy.dot(diff,numpy.transpose(diff))
            phi[i][j] = math.exp(temp * (-0.5))

    return means,varInv,phi

def GetSingleBasisFunc(mean,varInv,x):

    result = numpy.zeros(shape=(M))
    for i in range(M):
            curMu = mean[i]
            diff = numpy.subtract(x,curMu)
            temp = numpy.dot(diff,numpy.transpose(diff))
            result[i] = math.exp(temp * (-0.5))
    return result

def GetErms(w,xTest,yTest,mean,varInv):
    wtrans = numpy.transpose(w)
    rows,cols = yTest.shape
    diff = 0.0
    for i in range(rows):
        phi = GetSingleBasisFunc(mean,varInv,xTest[i])
        train = numpy.dot(wtrans,phi)
        diff += (yTest[i] - train) ** 2
    diff = diff/rows
    erms = math.sqrt(diff)
    
    return erms

def GetDiffInWeights(wNew,weights):
    rows = wNew.shape[0]
    diff = 0.0
    for i in range(rows):
        diff += (wNew[i] - weights[i]) ** 2
    return math.sqrt(diff)


def GradientDescent(weights,phi,xTrain, yTrain,mean,varInv,xTest,yTest, xVali, yVali):
    rows,cols = phi.shape
    numWeights = weights.shape[0]
    neta = 0.0001
    reg = 0
    eRMSPrev = 100
    wNew = numpy.ones(weights.shape)
    print ("========================================")
    for i in range(100):
            
        
        for k in range(rows):
              
          weights = wNew
          wTrans = numpy.transpose(weights)
          for j in range(numWeights):
                    
            curValue = numpy.dot(wTrans,phi[k])
            diff = yTrain[k] - float(curValue)
            
            
            sumTerm = (diff * phi[k][j])
            sumTerm -= reg * weights[j]
            wNew[j] = weights[j] + (neta * sumTerm)
            sumTerm = 0.0
        
        print ("Iteration %d\n"%i)    
        eRMS = GetErms(wNew,xTrain,yTrain,mean,varInv)
        print ("ERMS Train  = %f"%eRMS)
        eRMSVali = GetErms(wNew,xVali,yVali,mean,varInv)
        print ("ERMS Validation  = %f"%eRMSVali)
        eRMSTest = GetErms(wNew,xTest,yTest,mean,varInv)
        print ("ERMS Test  = %f"%eRMSTest)
        
        print ("\nWeights = ")
        print (weights.T)
        
        print ("\n========================================\n")
        
        
        
        if numpy.absolute(eRMSPrev -  eRMS) < 0.0001:
            print (" ===================Gradient Descent Ends =============== " )
            print (" Thanks for being here :) " )
            print ("ERMS Training Set = %f"%eRMS)
            eRMSVali = GetErms(weights,xVali,yVali,mean,varInv)
            print ("ERMS Validation Set= %f"%eRMSVali)
            eRMSTest = GetErms(weights,xTest,yTest,mean,varInv)
            print ("ERMS Testing Set = %f"%eRMSTest)
            
            print ("Final Learning rate = %f"%neta)
            
            
            
            break
        
        if eRMSPrev < eRMS:
            neta = neta / 2.0
      
        else:
            weights = wNew
    
            eRMSPrev = eRMS
          
    
        
def RegularizeLambda(phi,xTrain,yTrain,xTest,yTest,xVali,yVali):
    minERMSVali = 10000
    optLambda = -1
    regConst = 1
    for i in range(1,100):
        print ("For lambda = %d"%regConst)
        
        w = ClosedFormSolution(yTrain,phi,regConst)
    
        eRMSTrain = GetErms(w,xTrain,yTrain)
        print ("ERMS for train set = %f"%eRMSTrain)
            
        eRMSVali = GetErms(w,xVali,yVali)
        print ("ERMS for Validation set = %f"%eRMSVali)
            
        eRMSTest = GetErms(w,xTest,yTest)
        print ("ERMS for Test set = %f"%eRMSTest)
        
        
        if minERMSVali > eRMSVali:
            minERMSVali = eRMSVali
            optLambda = regConst
        
        regConst *= 2
            
        print ("Optimum Lambda Found Out = %d with Vali ERMS = %f"%(optLambda,minERMSVali))

def GetOptimumM():
    xTrain,yTrain,xTest,yTest,xVali,yVali = ProcessRawData()


    regConst = 0
    minRms = 100
    minRms = 100
    Mvals = numpy.zeros(shape=(50,1))
    avgERMS = numpy.zeros(shape=(50,1))
    for i in range(1,51):
        global M
        M=i
        Mvals[i-1] = i
        print ("For M = %d"%M)
        mean,varInv,phi = GetBasisFunction(xTrain)
        w = ClosedFormSolution(yTrain,phi,regConst)
        
        eRMSTrain = GetErms(w,xTrain,yTrain,mean,varInv)
        print ("ERMS for train set = %f"%eRMSTrain)
        
        eRMSVali = GetErms(w,xVali,yVali,mean,varInv)
        print ("ERMS for Validation set = %f"%eRMSVali)
        
        eRMSTest = GetErms(w,xTest,yTest,mean,varInv)
        print ("ERMS for Test set = %f"%eRMSTest)
        
        avgERMS[i-1] = (eRMSTrain + eRMSVali + eRMSTest) / 3.0
        
        if minRms > avgERMS[i-1]:
            optM = M
            mTrainRMS = eRMSTrain
            mValiRMS = eRMSVali
            mTestRMS = eRMSTest
            minRms = avgERMS[i-1]
        
            print ("Minim RMS For M = %f"%optM)
            print ("ERMS for Train set = %f"%mTrainRMS)
            print ("ERMS for Vali set = %f"%mValiRMS)
            print ("ERMS for Test set = %f"%mTestRMS)
        print (" Weights for closed form with no regularization = ")
        print (numpy.transpose(w))
        wInitial = numpy.zeros(shape=w.shape)
        accuracy = TestParams(w,mean,varInv,xTrain,yTrain)
        
        print ("=========================================")
        
        print ("Minim RMS For M = %f"%optM)
        print ("ERMS for Train set = %f"%mTrainRMS)
        print ("ERMS for Vali set = %f"%mValiRMS)
        print ("ERMS for Test set = %f"%mTestRMS)
    
    """pt.plot(Mvals, avgERMS)
    pt.xlabel('M basis functions')
    pt.ylabel('Avg E-RMS (Train + Test + Vali) / 2')
    pt.show()"""
    
        
        
    
def RegularizeLambda(phi,xTrain,yTrain,xTest,yTest,xVali,yVali):
    minERMSVali = 10000
    optLambda = -1
    regConst = 0.01
    eRMSValiArray = numpy.zeros(shape = (50,1))
    regConstArray = numpy.zeros(shape = (50,1))

    for i in range(1,51):
        print ("For lambda = %f"%regConst)
        
        w = ClosedFormSolution(yTrain,phi,regConst)
    
        eRMSTrain = GetErms(w,xTrain,yTrain,mean,varInv)
        print ("ERMS for train set = %f"%eRMSTrain)
            
        eRMSVali = GetErms(w,xVali,yVali,mean,varInv)
        print ("ERMS for Validation set = %f"%eRMSVali)
            
        eRMSTest = GetErms(w,xTest,yTest,mean,varInv)
        print ("ERMS for Test set = %f"%eRMSTest)
        
        eRMSValiArray[i-1] = eRMSVali
        regConstArray[i-1] = regConst
        
        if minERMSVali > eRMSVali:
            minERMSVali = eRMSVali
            optLambda = regConst
        
        regConst += 0.05
            
        print ("Optimum Lambda Found Out = %f with Vali ERMS = %f"%(optLambda,minERMSVali))
        
    print ("Reg Const Values")
    print (regConstArray)
    
    """pt.plot(regConstArray,eRMSValiArray)
    pt.xlabel(' Regularization Constant ' )
    pt.ylabel(' E Rms for Validation Set' )
    pt.show()"""
    
           
print ("Name : Vaibhav Sinha")
print ("UB Person # : 50208769")
print ("UD ID : vsinha2")
print ("\n\nLinear Regression on Synthetic Data Set\n\n")
print ("\nHyperparameters")
print ("\nM = 18 & Mu Found using K Means " )
print ("\nLambda = 0.26 for Closed Form Solution " )
print ("\nUsing Identity Matrix for Sigma " )

print (" ===================================================== " )
print (" ========== CLOSED FORM SOLUTION =====================30 Sec Wait!...THANKS :) " )  



xTrain,yTrain,xTest,yTest,xVali,yVali = ProcessRawData()
mean,varInv,phi = GetBasisFunction(xTrain)


regConst = 0.26
w = ClosedFormSolution(yTrain,phi,regConst)

print ( "\nWeights Found Out : " )
print (w.T)


eRMSTrain = GetErms(w,xTrain,yTrain,mean,varInv)
print ("\nERMS for Training set = %f"%eRMSTrain)
    
eRMSVali = GetErms(w,xVali,yVali,mean,varInv)
print ("\nERMS for Validation set = %f"%eRMSVali)
    
eRMSTest = GetErms(w,xTest,yTest,mean,varInv)
print ("\nERMS for Test set = %f"%eRMSTest)

print ("=============================================")

print (" Starting Gradient Descent " )
print (" Runs for about 93 iterations ( 2-3 mins . Thanks for your patience! ) ")


wInitial = numpy.zeros(shape=w.shape)

w = GradientDescent(wInitial,phi,xTrain,yTrain,mean,varInv,xTest,yTest,xVali,yVali)











