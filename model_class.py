import ACPF

class ACP_classique():
    #Initialisation d'une instance avec les paramètres de la méthode
    def __init__(self,n_pc,param):
        self.n_pc = n_pc
        self.param = param
    
    def train(self,x_train,y_train,verbose=False):
        self.models, self.V, self.y_bar = ACPF.ACP_train(x_train,y_train,self.n_pc,self.param,verbose)

    def predict(self,x_test):
        return ACPF.ACP_predict(self.models,x_test,self.n_pc,self.V,self.y_bar)
    
class ACPF_Ondelettes():
    def __init__(self,n_pc,param,p,J):
        self.n_pc = n_pc
        self.param = param
        self.p = p 
        self.J = J
    def train(self,x_train,y_train,verbose=False):
        self.models,self.V, self.y_bar, self.coeffs_wavelets_mean ,self.coeffs_shapes, self.signal_length ,self.indices_ACP, self.indices_mean = ACPF.Ondelettes_train(x_train,y_train,self.n_pc,self.param,verbose,0,self.p,self.J)
    
    def predict(self,x_test):
        return ACPF.Ondelettes_predict(self.models,x_test,self.n_pc,self.V,self.y_bar,self.coeffs_wavelets_mean,self.coeffs_shapes,self.signal_length,self.indices_ACP,self.indices_mean)
    
class ACPF_Bsplines():
    def __init__(self,n_pc,param,t1,t2,degree=1):
        self.n_pc = n_pc
        self.param = param
        self.degree = degree
        self.t1 = t1
        self.t2 = t2
    def train(self,x_train,y_train,verbose=False):
        self.models, self.V, self.y_bar, self.Bxy = ACPF.B_Splines_train(x_train,y_train,self.t1,self.t2,self.n_pc,self.param,verbose,self.degree)

    def predict(self,x_test):
        return ACPF.B_Splines_predict(self.models,x_test,self.n_pc,self.V,self.y_bar,self.Bxy)