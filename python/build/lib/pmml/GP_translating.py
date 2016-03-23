# This file is part of the PMML package for python.
#
# The PMML package is free software: you can redistribute it and/or 
# modify it.
#
# The PMML package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Please see the
######################################################################################
#
# Author: Engineering informatics group
# Date: Feb 2015
#-------------------------------------------------------------------------------------

import lxml.etree as ET
import numpy as np
import datetime
import getpass

def trans_get_para(model):

    X=model.X
    Y=model.y
    nugget=model.nugget
    k_lambda=model.theta_
    
    X=np.squeeze(X)
    Y=np.squeeze(Y)
    k_lambda=np.squeeze(k_lambda)
    gamma=1
    normalize=model.normalize
    corr=model.corr
    regr=model.regr
    beta=model.beta
    beta=np.squeeze(beta)

    return X,Y,nugget,k_lambda,gamma,normalize,corr,beta,regr

def trans_get_dimension(X,Y):

    sx=X.shape
    sy=Y.shape

    xrow=sx[0]
    yrow=sy[0]


    if len(sx)==1:
        xcol=1
    else:
        xcol=sx[1]
        
    if len(sy)==1:
        ycol=1
    else:
        ycol=sy[1]

    return xrow,yrow,xcol,ycol

def trans_name(xcol, ycol):

    featureName=[]
    for i in range(xcol):
        featureName.append('x{0}'.format(i+1))

    targetName=[]
    for i in range(ycol):
        targetName.append('y{0}'.format(i+1))

    return featureName,targetName

""" some basic information """
def trans_root(description,copyright,Annotation):
    
    time = datetime.datetime.now()
    time=str(time.year)+"-"+str(time.month)+"-"+str(time.day)+" "+str(time.hour)+":"+ str(time.minute)+":"+str(time.second)
    user_name=str(getpass.getuser())
    py_version="0.1"

    xmlns="http://www.dmg.org/PMML-4_2"
    xsi="http://www.w3.org/2001/XMLSchema-instance"
    schemaLocation="http://www.dmg.org/PMML-4_2 http://www.dmg.org/v4-2/pmml-4-2.xsd"
    PMML_version="4.2"
    ns = "{xsi}"

    """define namesapce"""
    PMML=root = ET.Element("{"+xmlns+"}PMML",\
                            nsmap={None:xmlns,'xsi':xsi},version=PMML_version,\
                            attrib={"{" + xsi + "}schemaLocation" : schemaLocation},)

    """pmml level"""
    if copyright is None:
        copyright="Copyright (c) 2015 {0}".format(user_name)
    if description is None:
        description="Gaussian Process Model"
    Header=ET.SubElement(PMML,"Header",copyright=copyright,\
                         description=description)

    """header level"""
    ET.SubElement(Header,"Extension", name="user",value=user_name,extender="Python-PMML")
    ET.SubElement(Header,"Application", name="Python-PMML",version=py_version)
    ET.SubElement(Header,"Timestamp").text=time
    
    if Annotation is not None:
        ET.SubElement(Header,"Annotation").text=Annotation
    return PMML
    
"""DataField level"""   
def trans_dataDictionary(PMML,featureName,targetName,xcol,ycol):
    DataDictionary=ET.SubElement(PMML,"DataDictionary",numberOfFields="{0}".format(xcol+ycol))

    for it_name in featureName:
        ET.SubElement(DataDictionary, "DataField", name=it_name,optype="continuous", dataType="double" )

    for it_name in targetName:
        ET.SubElement(DataDictionary, "DataField", name=it_name,optype="continuous", dataType="double" )

    return PMML

"""GP level"""        
def trans_GP(PMML):
   
    GaussianProcessModel=ET.SubElement(PMML,"GaussianProcessModel")
    GaussianProcessModel.set("modelName","Gaussian Process Model")
    GaussianProcessModel.set("functionName","regression")

    return GaussianProcessModel

"""minging schema"""
def tarns_minningSchema(GaussianProcessModel,featureName,targetName):

    MiningSchema=ET.SubElement(GaussianProcessModel,"MiningSchema")
    for it_name in featureName:
        ET.SubElement(MiningSchema, "MiningField", name=it_name,usageType="active")

    for it_name in targetName:
        ET.SubElement(MiningSchema, "MiningField", name=it_name,usageType="predicted")

    return GaussianProcessModel

"""Output"""
def trans_output(GaussianProcessModel):
    
    Output=ET.SubElement(GaussianProcessModel,"Output")
    ET.SubElement(Output,"OutputField",name="MeanValue",optype="continuous",dataType="double", feature="predictedValue")
    ET.SubElement(Output,"OutputField",name="StandardDeviation",optype="continuous",dataType="double", feature="predictedValue")
    return GaussianProcessModel

"""Derivate field"""
def trans_derivateField(GaussianProcessModel,featureName,targetName,model):
    derived_featureName=[]
    for i in range(len(featureName)):
        derived_featureName.append("derived_{0}".format(featureName[i]))
    derived_targetName=[]
    for i in range(len(targetName)):
        derived_targetName.append("derived_{0}".format(targetName[i]))
    if model.normalize :

        LocalTransformations=ET.SubElement(GaussianProcessModel,"LocalTransformations")
        m=model.X_mean.reshape(len(featureName),1)
        s=model.X_std.reshape(len(featureName),1)

        for i in range(len(featureName)):
            DerivedField=ET.SubElement(LocalTransformations,"DerivedField"\
                           ,name=derived_featureName[i],optype="continuous",dataType="double")
            NormContinuous=ET.SubElement(DerivedField,"NormContinuous"\
                                         ,field="{0}".format(featureName[i]))
            LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="0",norm="{0}".format(float(-m[i]/s[i])))
            LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="{0}".format(float(m[i])),norm="0")

        m=model.y_mean.reshape(len(targetName),1)
        s=model.y_std.reshape(len(targetName),1)
        for i in range(len(targetName)):
            DerivedField=ET.SubElement(LocalTransformations,"DerivedField",\
                           name=derived_targetName[i],optype="continuous",dataType="double")
            NormContinuous=ET.SubElement(DerivedField,"NormContinuous",\
                                         field="{0}".format(targetName[i]))
            LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="0",norm="{0}".format(float(-m[i]/s[i])))
            LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="{0}".format(float(m[i])),norm="0")

    return GaussianProcessModel,derived_featureName,derived_targetName

"""Kernel type"""
def trans_kernel(GaussianProcessModel,k_lambda,nugget,gamma,xcol,corr):

    lamb_inv=[np.sqrt(1.0/x) for x in k_lambda]
    theta="{0}".format(lamb_inv[0])
    for i in range(1,xcol):
        theta=theta+" {0}".format(lamb_inv[i])
        
    if "absolute_exponential" in str(corr):
        AbsoluteExponentialKernelType =ET.SubElement(GaussianProcessModel,\
                                                "GPAbsoluteExponentialKernelType")
        ET.SubElement(AbsoluteExponentialKernelType,"gamma",value="1")
        ET.SubElement(AbsoluteExponentialKernelType,"noiseVariance",value="{0}".format(nugget))                                        
        lambda_=ET.SubElement(AbsoluteExponentialKernelType,"lambda")
        ET.SubElement(lambda_,"Array",n="{0}".format(xcol),type="real").text=theta
        
    elif "squared_exponential" in str(corr):
        SquaredExponentialKernelType =ET.SubElement(GaussianProcessModel,\
                                                "GPSquaredExponentialKernelType")
        ET.SubElement(SquaredExponentialKernelType,"gamma",value="1")
        ET.SubElement(SquaredExponentialKernelType,"noiseVariance",value="{0}".format(nugget))                                        
        lambda_=ET.SubElement(SquaredExponentialKernelType,"lambda")
        ET.SubElement(lambda_,"Array",n="{0}".format(xcol),type="real").text=theta

    return GaussianProcessModel

def trans_regression(GaussianProcessModel,featureName,beta,regr):
    beta_index=0
    print(featureName,regr)  
    if "linear" in str(regr) or "quadratic" in str(regr):
        reg=ET.SubElement(GaussianProcessModel,"RegressionTable", intercept="{0}".format(beta[beta_index]))
        for it_name in featureName:
            beta_index+=1
            ET.SubElement(reg,"NumericPredictor",name="{0}".format(it_name),\
                          value="{0}".format(beta[beta_index]))
    if "quadratic" in str(regr):
        for i in range(len(featureName)):
            for j in range(i,len(featureName)):
                beta_index+=1
                pt=ET.SubElement(reg,"PredictorTerm",coefficient="{0}".format(beta[beta_index]))
                ET.SubElement(pt,"FieldRef",field="{0}".format(featureName[i]))
                ET.SubElement(pt,"FieldRef",field="{0}".format(featureName[j]))
    return GaussianProcessModel

"""GaussianProcessDictionary"""
def trans_GPDictionary(GaussianProcessModel,xrow):
    GaussianProcessDictionary =ET.SubElement(GaussianProcessModel,"GaussianProcessDictionary")
    GaussianProcessDictionary.set("numberOfSamples","{0}".format(xrow))

    return GaussianProcessDictionary

"""GaussianProcessFeature"""
def trans_GPD_feature(GaussianProcessDictionary,xcol,xrow,featureName,X,normalize,derived_featureName):   
    GaussianProcessFeature=ET.SubElement(GaussianProcessDictionary,"GaussianProcessFeature")
    FF=ET.SubElement(GaussianProcessFeature,"GaussianProcessFeatureFields",\
                        numberOfFields="{0}".format(xcol))
    if normalize is not True:
        for name in range(xcol):
            ET.SubElement(FF,"FieldRef",field="{0}".format(featureName[name]))
    else:
        for name in range(xcol):
            ET.SubElement(FF,"FieldRef",field="{0}".format(derived_featureName[name]))
        
    for row in range(xrow):
        print_row=row+1
        Instance=ET.SubElement(GaussianProcessFeature,"GaussianProcessFeatureInstance",\
                           id="{0}".format(print_row))
        
        Matrix=ET.SubElement(Instance,"REAL-SparseArray", n="{0}".format(xcol))
        
        index="{0}".format(1)
        value="{0}".format(X[row][0])
        for i in range(1,xcol):
            index=index+" {0}".format(i+1)
            value=value+" {0}".format(X[row][i])
            
        ET.SubElement(Matrix,"Indices").text=index
        ET.SubElement(Matrix,"REAL-Entries").text=value
 
"""GaussianProcessTarget"""    
def trans_GPD_target(GaussianProcessDictionary,ycol,yrow,targetName,Y,normalize,derived_targetName):

    GaussianProcessTarget=ET.SubElement(GaussianProcessDictionary,"GaussianProcessTarget")
    TF=ET.SubElement(GaussianProcessTarget,"GaussianProcessTargetFields",\
                        numberOfFields="{0}".format(ycol))
    if normalize is not True:
        for name in range(ycol):
            ET.SubElement(TF,"FieldRef",field="{0}".format(targetName[name]))
    else:
        for name in range(ycol):
            ET.SubElement(TF,"FieldRef",field="{0}".format(derived_targetName[name]))


       
    for row in range(yrow):
        print_row=row+1
        Instance=ET.SubElement(GaussianProcessTarget,"GaussianProcessTargetInstance",\
                           id="{0}".format(print_row))
        
        Matrix=ET.SubElement(Instance,"REAL-SparseArray", n="{0}".format(ycol))
        
        index="{0}".format(1)
        value="{0}".format(Y[row])
        for i in range(1,ycol):
            index=index+" {0}".format(i+1)
            value=value+" {0}".format(Y[row])
            
        ET.SubElement(Matrix,"Indices").text=index
        ET.SubElement(Matrix,"REAL-Entries").text=value
