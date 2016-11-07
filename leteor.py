import numpy
import os.path
import math
from sklearn.cluster import KMeans


global M
M=10

def ProcessRawData():
    train = numpy.loadtxt('Querylevelnorm.txt', dtype=str, delimiter=' ')
    test = numpy.loadtxt('Querylevelnorm.txt', dtype=str, delimiter=' ')
    vali = numpy.loadtxt('Querylevelnorm.txt', dtype=str, delimiter=' ')
    
    trainRows,trainCols = train.shape
    testRowsLast = int(trainRows * 0.9)
        
    trainRows = trainRows * 0.8
    trainRows = int(int(trainRows / M) * M)
        
    train = train[:trainRows,:]
    
    test = test[trainRows:testRowsLast,:]
    
    vali = vali[testRowsLast:,:]

    xTrain = GetX(train)
    yTrain = GetY(train)

    xTest = GetX(test)
    yTest = GetY(test)

    xVali = GetX(vali)
    yVali = GetY(vali)

    numpy.savetxt('xTrain.txt', xTrain)
    numpy.savetxt('yTrain.txt', yTrain)

    numpy.savetxt('xTest.txt', xTest)
    numpy.savetxt('yTest.txt', yTest)
    
    numpy.savetxt('xVali.txt', xVali)
    numpy.savetxt('yVali.txt', yVali)

    return xTrain,yTrain,xTest,yTest,xVali,yVali

def GetX(data):
    rows,cols = data.shape
    result = numpy.zeros(shape=(rows,46))
    
    x = data[0:rows,2:48]
    for i in range(rows):
        for j in range(46):
            index = x[i][j].index(':')
            x[i][j] = x[i][j][index+1:]
            result[i][j] = float(x[i][j])

    return result

def GetY(data):
    rows,cols = data.shape
    result = numpy.zeros(shape=(rows,1))
    y = data[0:rows,:]
    for i in range(0,rows):
        result[i][0] = float(y[i][0])
    return result

def GetProcessedY(data):
    rows = data.shape[0]
    result = numpy.zeros(shape=(rows,1))
    for i in range(0,rows):
        result[i][0] = float(data[i])
    return result


def GetProcessedData():
    xTrain = numpy.loadtxt('xTrain.txt', dtype=float)
    yTrain = numpy.loadtxt('yTrain.txt', dtype=float)
    yTrain = GetProcessedY(yTrain)

    xTest = numpy.loadtxt('xTest.txt', dtype=float)
    yTest = numpy.loadtxt('yTest.txt', dtype=float)
    yTest = GetProcessedY(yTest)


    xVali = numpy.loadtxt('xVali.txt', dtype=float)
    yVali = numpy.loadtxt('yVali.txt', dtype=float)
    yVali = GetProcessedY(yVali)
    
    return xTrain,yTrain,xTest,yTest,xVali,yVali


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
    
    means = numpy.zeros(shape=(M,46))
    
    kmeans = KMeans(n_clusters=M, random_state=0).fit(xTrain)
    means =  kmeans.cluster_centers_
        
    variance = numpy.var(x,axis=0)
    
    varianceMatrix = numpy.zeros(shape=(46,46))
    for i in range(46):
        varianceMatrix[i][i] = (variance[i]/10)
        if(variance[i] == 0) :
            varianceMatrix[i][i] = 0.001
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
    accuracy = TestParams(weights,mean,varInv,xTest,yTest)
    neta = 0.00001
    reg = 0
    eRMSPrev = 100
    print ("========================================")
    wNew = numpy.ones(weights.shape)
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
        
        print ("Iteration %d"%i)  
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
            print ("ERMS Train Final = %f"%eRMS)
            eRMSVali = GetErms(weights,xVali,yVali,mean,varInv)
            print ("ERMS Validation Final= %f"%eRMSVali)
            eRMSTest = GetErms(weights,xTest,yTest,mean,varInv)
            print ("ERMS Test Final= %f"%eRMSTest)
            
            print ("Final Learning rate = %f"%neta)
            
            break
        
        
        
        if eRMSPrev < eRMS:
            neta = neta / 2.0
       
    
        else:
            weights = wNew
    
            eRMSPrev = eRMS
          
              


def GetOptimumM():
    if(os.path.isfile('xTrain.txt') and os.path.isfile('yTrain.txt') and os.path.isfile('xTest.txt') and os.path.isfile('yTest.txt')
    and os.path.isfile('xVali.txt') and os.path.isfile('yVali.txt')):
        xTrain,yTrain,xTest,yTest,xVali,yVali = GetProcessedData()
    else:
        xTrain,yTrain,xTest,yTest,xVali,yVali = ProcessRawData()


    regConst = 0
    minRms = 100
    minRms = 100
    avgERMS = numpy.zeros(shape=(46,1))
    Mvals = numpy.zeros(shape=(46,1))
    for i in range(1,46):
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
        
        avgERMS[i] = (eRMSTrain + eRMSVali + eRMSTest) / 3.0
        
        if minRms > avgERMS[i]:
            optM = M
            mTrainRMS = eRMSTrain
            mValiRMS = eRMSVali
            mTestRMS = eRMSTest
            minRms = avgERMS[i]
        
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
    

    pt.plot(Mvals,avgERMS)
    #pt.xlabel('M basis functions')
    pt.ylabel('Avg E-RMS (Train + Test + Vali) / 2')
    pt.show()
   
        
        
    
def RegularizeLambda(phi,xTrain,yTrain,xTest,yTest,xVali,yVali):
    minERMSVali = 10000
    optLambda = -1
    regConst = 0.01
    eRMSValiArray = numpy.zeros(shape = (50,1))
    regConstArray = numpy.zeros(shape = (50,1))

    for i in range(1,25):
        print ("For lambda = %f"%regConst)
        
        w = ClosedFormSolution(yTrain,phi,regConst)
    
        eRMSTrain = GetErms(w,xTrain,yTrain,mean,varInv)
        print ("ERMS for train set = %f"%eRMSTrain)
            
        eRMSVali = GetErms(w,xVali,yVali,mean,varInv)
        print ("ERMS for Validation set = %f"%eRMSVali)
            
        eRMSTest = GetErms(w,xTest,yTest,mean,varInv)
        print ("ERMS for Test set = %f"%eRMSTest)
        
        eRMSValiArray[i] = eRMSVali
        regConstArray[i] = regConst
        
        if minERMSVali > eRMSVali:
            minERMSVali = eRMSVali
            optLambda = regConst
        
        regConst += 0.05
            
        print ("Optimum Lambda Found Out = %f with Vali ERMS = %f"%(optLambda,minERMSVali))
        
    print ("Reg Const Values")
  
    pt.plot(regConstArray,eRMSValiArray)
    pt.xlabel(' Regularization Constant ' )
    pt.ylabel(' E Rms for Validation Set' )
    pt.show()
    
    
    

print ("Name : Vaibhav Sinha")
print ("UB Person # : 50208769")
print ("UD ID : vsinha2")
print ("\nLinear Regression on LETOR Data Set\n")
print ("Hyperparameters")
print ("M = 10 & Mu Found using K Means " )
print ("Lambda = 0.0 for Closed Form and Stochastic Gradient " )
print ("Using Identity Matrix for Sigma " )

print (" ===================================================== " )
print (" ========== CLOSED FORM SOLUTION ===================== 30 Sec Wait. THANKS.. :) " ) 

    
    
    
    
if(os.path.isfile('xTrain.txt') and os.path.isfile('yTrain.txt') and os.path.isfile('xTest.txt') and os.path.isfile('yTest.txt')
    and os.path.isfile('xVali.txt') and os.path.isfile('yVali.txt')):
    xTrain,yTrain,xTest,yTest,xVali,yVali = GetProcessedData()
else:
    xTrain,yTrain,xTest,yTest,xVali,yVali = ProcessRawData()


mean,varInv,phi = GetBasisFunction(xTrain)




regConst = 0.26
w = ClosedFormSolution(yTrain,phi,regConst)

print ( "Weights Found Out : " )
print (w.T)


eRMSTrain = GetErms(w,xTrain,yTrain,mean,varInv)
print ("\nERMS for Training set = %f"%eRMSTrain)
    
eRMSVali = GetErms(w,xVali,yVali,mean,varInv)
print ("\nERMS for Validation set = %f"%eRMSVali)
    
eRMSTest = GetErms(w,xTest,yTest,mean,varInv)
print ("\nERMS for Test set = %f"%eRMSTest)

print ("=============================================")

print (" Starting Gradient Descent " )
print (" Runs for about 55 iterations ( 2-3 mins . Thanks for your patience! ) ")


wInitial = numpy.ones(shape=w.shape)


w = GradientDescent(wInitial,phi,xTrain,yTrain,mean,varInv,xTest,yTest,xVali,yVali)











