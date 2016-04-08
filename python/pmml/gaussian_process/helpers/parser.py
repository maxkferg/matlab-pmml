import lxml.etree as ET
import numpy as np

def parse_GPM(nsp,filename):
    """Return the PMML document as an etree element"""
    tree = ET.parse(filename)
    root = tree.getroot()
    tagname = "GaussianProcessModel".lower()
    GPM = root.find(nsp + tagname)
    if GPM is None:
        raise "Missing tag %s"%tagname
    return GPM



def parse_name(nsp,GPM):
    """parse MiningSchema for features and targets"""
    # Will get a list of target name and feature name
    tagname = "MiningSchema".lower();
    MS=GPM.find(nsp+tagname)
    targetName=[]
    featureName=[]
    for MF in MS:
        MF_name=MF.attrib["name"]
        MF_type=MF.attrib["usagetype"]
        if MF_type == "active":
            featureName.append(MF_name)
        elif MF_type == "predicted":
            targetName.append(MF_name)

    return featureName,targetName



def parse_kernel(nsp,GPM):
    """Return kernel parameters"""
    kernelName = None;

    name = "ARDSquaredExponentialKernel".lower()
    kernel = GPM.find(nsp+name)
    if kernel is not None:
        kernelName = "ARDSquaredExponentialKernelType"
        nugget = float(kernel.attrib["noisevariance"])
        gamma = float(kernel.attrib["gamma"])
        array = kernel.find(nsp+"lambda").find(nsp+"array").text
        array = array.strip()
        k_lambda = [float(i) for i in array.split(" ")]


    name = "AbsoluteExponentialKernelType".lower()
    kernel = GPM.find(nsp+name)
    if kernel is not None:
        kernelName = "AbsoluteExponentialKernelType"
        array = kernel.find(nsp+"lambda").find(nsp+"array").text
        array = array.strip()
        k_lambda = [float(i) for i in array.split(" ")]
        nugget = float(kernel.attrib["noisevariance"])
        gamma = float(kernel.attrib["gamma"])

    if kernelName is None:
        raise "Unable to find valid kernel tag"

    return kernelName,k_lambda,nugget,gamma



def parse_training_values(nsp,GPM):
    """Return the training values"""
    traininginstances = GPM.find(nsp+"traininginstances")
    inlinetable = traininginstances.find(nsp+"inlinetable")
    instancefields = traininginstances.find(nsp+"instancefields")

    [features,targets] = parse_name(nsp,GPM)

    nrows = int(traininginstances.attrib['recordcount'])
    fcols = len(features)
    tcols = len(targets)

    xTrain = np.zeros([nrows,fcols]);
    yTrain = np.zeros([nrows,tcols]);

    for i,row in enumerate(inlinetable.findall(nsp+"row")):
        for j,featureName in enumerate(features):
            col = row.find(nsp+featureName)
            xTrain[i][j] = float(col.text)

        for j,featureName in enumerate(targets):
            col = row.find(nsp+featureName)
            yTrain[i][j] = float(col.text)

    return xTrain,yTrain


