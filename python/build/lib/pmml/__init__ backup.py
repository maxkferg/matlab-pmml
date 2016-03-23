import lxml.etree as ET
import lxml.builder
import numpy as np
import datetime
import getpass

class GP:

    def __init__(self):
        self.X=[]
        self.Y=[]
        self.X_mean=[]
        self.X_std=[]
        self.Y_mean=[]
        self.Y_std=[]
        self.k_lambda=[]

        self.gamma=0
        self.nugget=0
        self.kernelName=0


    def GP_translator(self,file_name,model,project_name=None):

        """name for collumns of X,y"""
        X=model.X
        Y=model.y
        nugget=model.nugget
        k_lambda=model.theta_
        
        X=np.squeeze(X)
        Y=np.squeeze(Y)
        k_lambda=np.squeeze(k_lambda)

        """define demension"""
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

        featureName=[]
        for i in range(xcol):
            featureName.append('x{0}'.format(i+1))

        targetName=[]
        for i in range(ycol):
            targetName.append('y{0}'.format(i+1))

        X=np.array(X)
        Y=np.array(Y)

        """ some basic information """
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
        Header=ET.SubElement(PMML,"Header",copyright="Copyright (c) 2015 {0}".format(user_name),\
                             description="Gaussian Process Model")
        DataDictionary=ET.SubElement(PMML,"DataDictionary",numberOfFields="{0}".format(xcol+ycol))
        GaussianProcessModel=ET.SubElement(PMML,"GaussianProcessModel")

        """header level"""
        ET.SubElement(Header,"Extension", name="user",value=user_name,extender="Python-PMML")
        ET.SubElement(Header,"Application", name="Python-PMML",version=py_version)
        ET.SubElement(Header,"Timestamp").text=time
        
        if project_name is not None:
            ET.SubElement(Header,"Project", name=project_name)

        """DataField level"""
        for it_name in featureName:
            ET.SubElement(DataDictionary, "DataField", name=it_name,optype="continuous", dataType="double" )

        for it_name in targetName:
            ET.SubElement(DataDictionary, "DataField", name=it_name,optype="continuous", dataType="double" )
            
        """GP level"""
        GaussianProcessModel.set("modelName","Gaussian Process Model")
        GaussianProcessModel.set("functionName","Regression")


        """minging schema"""
        MiningSchema=ET.SubElement(GaussianProcessModel,"MiningSchema")
        for it_name in featureName:
            ET.SubElement(MiningSchema, "MiningField", name=it_name,usageType="active")

        for it_name in targetName:
            ET.SubElement(MiningSchema, "MiningField", name=it_name,usageType="predicted")


        """Output"""
        Output=ET.SubElement(GaussianProcessModel,"Output")
        ET.SubElement(Output,"OutputField",name="MeanValue",optype="continuous",dataType="double", feature="predictedValue")
        ET.SubElement(Output,"OutputField",name="StandardDeviation",optype="continuous",dataType="double", feature="predictedValue")


        """Derivate field"""
        if model.normalize :
            derived_featureName=[]
            for i in range(len(featureName)):
                derived_featureName.append("derived_{0}".format(featureName[i]))
            derived_targetName=[]
            for i in range(len(targetName)):
                derived_targetName.append("derived_{0}".format(targetName[i]))


            LocalTransformations=ET.SubElement(GaussianProcessModel,"LocalTransformations")
            m=model.X_mean.reshape(len(featureName),1)
            s=model.X_std.reshape(len(featureName),1)

            for i in range(len(featureName)):
                DerivedField=ET.SubElement(LocalTransformations,"DerivedField"\
                               ,name=derived_featureName[i],optype="continuous",dataType="double")
                NormContinuous=ET.SubElement(DerivedField,"NormContinuous"\
                                             ,feild="{0}".format(featureName[i]))
                LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="0",norm="{0}".format(float(-m[i]/s[i])))
                LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="{0}".format(float(m[i])),norm="0")

            m=model.y_mean.reshape(len(targetName),1)
            s=model.y_std.reshape(len(targetName),1)
            for i in range(len(targetName)):
                DerivedField=ET.SubElement(LocalTransformations,"DerivedField",\
                               name=derived_targetName[i],optype="continuous",dataType="double")
                NormContinuous=ET.SubElement(DerivedField,"NormContinuous",\
                                             feild="{0}".format(targetName[i]))
                LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="0",norm="{0}".format(float(-m[i]/s[i])))
                LinearNorm=ET.SubElement(NormContinuous,"LinearNorm",orig="{0}".format(float(m[i])),norm="0")


        """Kernel type"""

        theta="{0}".format(k_lambda[0])
        for i in range(1,xcol):
            theta=theta+" {0}".format(k_lambda[i])
            
        if "absolute_exponential" in str(model.corr):
            AbsoluteExponentialKernelType =ET.SubElement(GaussianProcessModel,\
                                                    "AbsoluteExponentialKernelType",\
                                                    noise="{0}".format(nugget),\
                                                    gamma="1")
            AbsoluteExponentialKernelType.set("lambda",theta)
        elif "squared_exponential" in str(model.corr):
            SquaredExponentialKernelType =ET.SubElement(GaussianProcessModel,\
                                                    "SquaredExponentialKernelType",\
                                                    noise="{0}".format(nugget),\
                                                    gamma="1" )
            SquaredExponentialKernelType.set("lambda",theta)

        elif "cubic" in str(model.corr):
            CubicKernelType =ET.SubElement(GaussianProcessModel,\
                                                    "CubicKernelType",\
                                                    noise="{0}".format(nugget),\
                                                    gamma="1")
            CubicKernelType.set("lambda",theta)
        elif "linear" in str(model.corr):
            LinearKernelType =ET.SubElement(GaussianProcessModel,\
                                                    "LinearKernelType",\
                                                    noise="{0}".format(nugget),\
                                                    gamma="1" )
            LinearKernelType.set("lambda",theta)

        """GaussianProcessDictionary"""
        GaussianProcessDictionary =ET.SubElement(GaussianProcessModel,"GaussianProcessDictionary")
        GaussianProcessDictionary.set("numberOfSamples","{0}".format(xrow))

        """dataDictionary level"""
        GaussianProcessFeature=ET.SubElement(GaussianProcessDictionary,"GaussianProcessFeature")
        GaussianProcessTarget=ET.SubElement(GaussianProcessDictionary,"GaussianProcessTarget")

        """GaussianProcessFeature"""
        FF=ET.SubElement(GaussianProcessFeature,"GaussianProcessFeatureFields",\
                            numberOfFields="{0}".format(xcol))
        if model.normalize is not True:
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
        TF=ET.SubElement(GaussianProcessTarget,"GaussianProcessTargetFields",\
                            numberOfFields="{0}".format(ycol))
        if model.normalize is not True:
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

        tree = ET.ElementTree(PMML)
        tree.write(file_name,pretty_print=True,xml_declaration=True )
    #------------------------------------- 
    def GP_parser(self,file_name):
        nsp="{http://www.dmg.org/PMML-4_2}" 
        tree = ET.parse(file_name)
        root = tree.getroot()

        """print out and check"""
        print root.tag,root.attrib
        for child in root:
            print child.tag
            
        """parse DataDictionary"""

        """parse GP"""
        GPM=root.find(nsp+"GaussianProcessModel")
        #----------------------------------------------------------

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
                
        #parse local transformation
        #----------------------------------------------------
        X_mean=np.ones(len(featureName))
        X_std=np.ones(len(featureName))
        Y_mean=np.ones(len(targetName))
        Y_std=np.ones(len(targetName))

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
                if i<len(featureName):
                    X_mean[i]=m
                    X_std[i]=s
                else:
                    Y_mean[i-len(featureName)]=m
                    Y_std[i-len(featureName)]=s
                i=i+1;
                    
        #parse Kernel
        #-------------------------------------------------
        kenelName="Nothing"

        kernel=GPM.find(nsp+"SquaredExponentialKernelType")
        if kernel is not None:
            k_lambda=[]
            kernelName="SquaredExponentialKernelType"
            array=kernel.attrib["lambda"]
            k_lambda=[float(i) for i in array.split(" ")]
            nugget=float(kernel.attrib["noise"])
            gamma=float(kernel.attrib["gamma"])
            
        kernel=GPM.find(nsp+"AbsoluteExponentialKernelType")
        if kernel is not None:
            k_lambda=[]
            kernelName="AbsoluteExponentialKernelType"
            array=kernel.attrib["lambda"]
            k_lambda=[float(i) for i in array.split(" ")]
            nugget=float(kernel.attrib["noise"])
            gamma=float(kernel.attrib["gamma"])
            
        kernel=GPM.find(nsp+"CubicKernelType")
        if kernel is not None:
            k_lambda=[]
            kernelName="CubicKernelType"
            array=kernel.attrib["lambda"]
            k_lambda=[float(i) for i in array.split(" ")]
            nugget=float(kernel.attrib["noise"])
            gamma=float(kernel.attrib["gamma"])

        kernel=GPM.find(nsp+"LinearKernelType")
        if kernel is not None:
            k_lambda=[]
            kernelName="LinearKernelType"
            array=kernel.attrib["lambda"]
            k_lambda=[float(i) for i in array.split(" ")]
            nugget=float(kernel.attrib["noise"])
            gamma=float(kernel.attrib["gamma"])
            
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

        self.X=np.array(X)
        self.Y=np.array(Y)
        self.X_mean=np.array(X_mean)
        self.X_std=np.array(X_std)
        self.Y_mean=np.array(Y_mean)
        self.Y_std=np.array(Y_std)
        self.k_lambda=np.array(k_lambda)

        self.gamma=gamma
        self.nugget=nugget
        self.kernelName=kernelName
        
        return self
