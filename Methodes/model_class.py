import Methodes.ACPF as ACPF

class ACP_classique():
    #Initialisation d'une instance avec les paramètres de la méthode
    def __init__(self,n_pc,param):
        self.n_pc = n_pc
        self.param = param
    
    def train(self,x_train,y_train,kernel_fn=None,verbose=False):
        self.models, self.V, self.y_bar = ACPF.ACP_train(x_train,y_train,self.n_pc,self.param,kernel_fn=None,verbose=verbose)

    def predict(self,x_test):
        return ACPF.ACP_predict(self.models,x_test,self.n_pc,self.V,self.y_bar)
    
class ACPF_Ondelettes():
    def __init__(self,n_pc,param,p,J):
        self.n_pc = n_pc
        self.param = param
        self.p = p 
        self.J = J
    def train(self,x_train,y_train,kernel_fn=None,verbose=False):
        self.models,self.V, self.y_bar, self.coeffs_wavelets_mean ,self.coeffs_shapes, self.signal_length ,self.indices_ACP, self.indices_mean = ACPF.Ondelettes_train(x_train,y_train,self.n_pc,self.param,0,self.p,self.J,kernel_fn=None,verbose=verbose)
    
    def predict(self,x_test):
        return ACPF.Ondelettes_predict(self.models,x_test,self.n_pc,self.V,self.y_bar,self.coeffs_wavelets_mean,self.coeffs_shapes,self.signal_length,self.indices_ACP,self.indices_mean)

class ACPF_Ondelettes2D():
    def __init__(self,n_pc,param,p,J):
        self.n_pc = n_pc
        self.param = param
        self.p = p 
        self.J = J
    def train(self,x_train,y_train,kernel_fn=None,verbose=False):
        self.models,self.V, self.y_bar, self.coeffs_wavelets_mean ,self.coeffs_slices, self.image_shape,self.indices_ACP, self.indices_mean = ACPF.Ondelettes2D_train(x_train,y_train,self.n_pc,self.param,0,self.p,self.J,kernel_fn=None,verbose=verbose)
    
    def predict(self,x_test):
        return ACPF.Ondelettes2D_predict(self.models,x_test,self.n_pc,self.V,self.y_bar,self.coeffs_wavelets_mean,self.coeffs_slices,self.image_shape,self.indices_ACP,self.indices_mean)
    
class ACPF_Bsplines():
    def __init__(self,n_pc,param,noeuds,domaine,degree=1):
        self.n_pc = n_pc
        self.param = param
        self.degree = degree
        self.noeuds = noeuds
        self.domaine=domaine
        
    def train(self,x_train,y_train,kernel_fn=None,verbose=False):
        self.models, self.V, self.y_bar, self.Bxy = ACPF.B_Splines_train(x_train,y_train,self.noeuds,self.domaine,self.n_pc,self.param,self.degree,kernel_fn=None,  verbose=verbose)

    def predict(self,x_test):
        return ACPF.B_Splines_predict(self.models,x_test,self.n_pc,self.V,self.y_bar,self.Bxy)