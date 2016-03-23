import lxml.etree as ET
import numpy as np
from . import GP_scoring as scoring
from . import GP_parsing as parsing
from . import GP_translating as translating

class GP:

    def __init__(self,description=None,copyright=None,Annotation=None):
        self.X=[]
        self.Y=[]
        self.X_mean=[]
        self.X_std=[]
        self.Y_mean=[]
        self.Y_std=[]
        self.k_lambda=[]
        self.beta=[]
        self.gamma=0
        self.nugget=0
        self.kernelName=0
        self.description=description
        self.copyright=copyright
        self.Annotation=Annotation


    #---------------------------------------------------------#
    #----------------write a model into a PMML file-----------#
    #---------------------------------------------------------#
    def GP_translator(self,file_name,model):

        X,Y,nugget,k_lambda,gamma,normalize,corr,beta,regr=translating.trans_get_para(model)

        xrow,yrow,xcol,ycol=translating.trans_get_dimension(X,Y)

        featureName,targetName=translating.trans_name(xcol, ycol)

        PMML=translating.trans_root(self.description,self.copyright,self.Annotation)

        PMML=translating.trans_dataDictionary(PMML,featureName,targetName,xcol,ycol)

        GaussianProcessModel=translating.trans_GP(PMML)

        GaussianProcessModel=translating.tarns_minningSchema(GaussianProcessModel,featureName,targetName)

        GaussianProcessModel=translating.trans_output(GaussianProcessModel)

        aussianProcessModel,derived_featureName,derived_targetName=translating.trans_derivateField(GaussianProcessModel,featureName,targetName,model)

        GaussianProcessModel=translating.trans_kernel(GaussianProcessModel,k_lambda,nugget,gamma,xcol,corr)

        GaussianProcessModel=translating.trans_regression(GaussianProcessModel,featureName,beta,regr)

        GaussianProcessDictionary=translating.trans_GPDictionary(GaussianProcessModel,xrow)

        translating.trans_GPD_feature(GaussianProcessDictionary,xcol,xrow,featureName,X,normalize,derived_featureName)

        translating.trans_GPD_target(GaussianProcessDictionary,ycol,yrow,targetName,Y,normalize,derived_targetName)

        tree = ET.ElementTree(PMML)

        tree.write(file_name,pretty_print=True,xml_declaration=True )

        tree.write(file_name,pretty_print=True,xml_declaration=True )
    #---------------------------------------------------------#
    #----------------Parse a PMML file ----- -----------------#
    #---------------------------------------------------------#
    def GP_parser(self,file_name):

        nsp="{http://www.dmg.org/PMML-4_2}"

        GPM=parsing.parse_GPM(nsp,file_name)

        featureName,targetName=parsing.parse_name(nsp,GPM)

        X_mean,X_std,Y_mean,Y_std=parsing.parse_mean_std(nsp,GPM,len(featureName),len(targetName))

        kernelName,k_lambda,nugget,gamma=parsing.parse_kernel(nsp,GPM)

        X,Y=parsing.parse_coeff(nsp,GPM)

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
    #---------------------------------------------------------#
    #---------------- New prediction   -----------------------#
    #---------------------------------------------------------#
    def GP_scorer(self,tx):

        # convert to lambda to theta (theta=(1/lambda)^2)
        theta=[(1/x)*(1/x) for x in self.k_lambda]
        #print theta
        #
        tx=(tx-self.X_mean)/self.X_std

        K_1=scoring.radi_k(self.X,theta,self.nugget,self.kernelName)
        k_T=scoring.radi_new_k(tx,self.X,theta,self.kernelName)

        p=np.dot(k_T,K_1)
        mu=np.dot(p,self.Y)

        #
        mu=(mu)*self.Y_std+self.Y_mean

        #normalized sigma?
        k_tx=scoring.radi_new_x(tx,theta,self.nugget,self.kernelName)
        t=k_tx-np.sum(p*k_T,axis=1)
        sigma=(t)

        return np.squeeze(mu),np.squeeze(sigma)