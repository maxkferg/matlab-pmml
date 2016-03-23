import lxml.etree as ET
import numpy as np
 
#------------------------------------- 
def parse_GPM(nsp,file_name):
    #nsp="{http://www.dmg.org/PMML-4_2}" 
    tree = ET.parse(file_name)
    root = tree.getroot()

    """parse GP"""
    GPM=root.find(nsp+"GaussianProcessModel")

    return GPM
    #----------------------------------------------------------
def parse_name(nsp,GPM):
    """parse MiningSchema"""
    #Will get a list of target name and feature name
    MS=GPM.find(nsp+'MiningSchema')
    targetName=[]
    featureName=[]
    for MF in MS:
        MF_name=MF.attrib["name"]
        MF_type=MF.attrib["usageType"]
        if MF_type == "active":
            featureName.append(MF_name)
        elif MF_type == "predicted":
            targetName.append(MF_name)
            
    return featureName,targetName
    #parse local transformation
    #----------------------------------------------------
def parse_mean_std(nsp, GPM,len_feature,len_target):
    X_mean=np.zeros(len_feature)
    X_std=np.ones(len_feature)
    Y_mean=np.zeros(len_target)
    Y_std=np.ones(len_target)

    LT=GPM.find(nsp+"LocalTransformations")

    if LT is not None:
        i=0
        for DF in LT.findall(nsp+"DerivedField"):
            NC=DF.find(nsp+"NormContinuous")
            norm=np.zeros(2)
            orig=np.zeros(2)
            j=0
            for LN in NC.findall(nsp+"LinearNorm"):
                norm[j]=LN.attrib["norm"]
                orig[j]=LN.attrib["orig"]
                j=j+1
                
            m=orig[1]
            s=float(-m/norm[0])
            if i<len_feature:
                X_mean[i]=m
                X_std[i]=s
            else:
                Y_mean[i-len_feature]=m
                Y_std[i-len_feature]=s
            i=i+1;
                
    return X_mean,X_std,Y_mean,Y_std

def parse_kernel(nsp,GPM):
    #parse Kernel
    #-------------------------------------------------
    kenelName="Nothing"

    kernel=GPM.find(nsp+"GPSquaredExponentialKernelType")
    if kernel is not None:
        kernelName="SquaredExponentialKernelType"
        nugget=float(kernel.find(nsp+"noiseVariance").attrib["value"])
        gamma=float(kernel.find(nsp+"gamma").attrib["value"])
        array=kernel.find(nsp+"lambda").find(nsp+"Array").text
        array = array.strip()
        k_lambda=[float(i) for i in array.split(" ")]
        
        
    kernel=GPM.find(nsp+"GPAbsoluteExponentialKernelType")
    if kernel is not None:
        kernelName="AbsoluteExponentialKernelType"
        array=kernel.find(nsp+"lambda").find(nsp+"Array").text
        array = array.strip()
        k_lambda=[float(i) for i in array.split(" ")]
        nugget=float(kernel.find(nsp+"noiseVariance").attrib["value"])
        gamma=float(kernel.find(nsp+"gamma").attrib["value"])

    return kernelName,k_lambda,nugget,gamma


def parse_coeff(nsp,GPM):    
    #parse coefficient 
    #-------------------------------------------------
    #for GP: the coefficient is the X and Y
    Coeff=GPM.find(nsp+"GaussianProcessDictionary")
    #parse feature (for multidimensional target)
    X=[]
    Feature=Coeff.find(nsp+"GaussianProcessFeature")
    for child in Feature.findall(nsp+"GaussianProcessFeatureInstance"):
        array=child.find(nsp+"REAL-SparseArray")
        array_string=array.find(nsp+"REAL-Entries").text
        array_number=[float(i) for i in array_string.split(' ')]
        X.append(array_number)
        

    #parse target (for one dimensional target)
    Y=[]
    Target=Coeff.find(nsp+"GaussianProcessTarget")
    for child in Target.findall(nsp+"GaussianProcessTargetInstance"):
        array=child.find(nsp+"REAL-SparseArray")
        array_string=array.find(nsp+"REAL-Entries").text
        Y.append(float(array_string))
    return X,Y


