import torch 
import torch.nn as nn
import torch.optim as optim


#SimpleNetwork has only one layer uses the Adam  Optimizer  

class SimpleNetwork(nn.Module):
    
    def __init__(self, input_dim, fc1_dim, fc2_dim,fc3_dim , fc4_dim, n_action, lr,loss):
        super(SimpleNetwork, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(

              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, n_action),
              
      

        )
 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions



class MoreLayersNetwork(nn.Module):
    
    def __init__(self, input_dim, fc1_dim, fc2_dim,fc3_dim , fc4_dim, n_action, lr,loss):
        super(MoreLayersNetwork, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(

              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, fc2_dim),
              nn.ReLU(),
              nn.Linear(fc2_dim, fc3_dim),
              nn.ReLU(),
              nn.Linear(fc3_dim, n_action),

        )
 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions
    
#we changed only the optimizer type  

class SimpleNetworkWithDiffrentOptimizer(nn.Module):
    
    def __init__(self, input_dim, fc1_dim, fc2_dim,fc3_dim , fc4_dim, n_action, lr,loss):
        super(SimpleNetworkWithDiffrentOptimizer, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(

              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, n_action),
        )
 
        #optimizer with addagrad  
        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions
    


class MoreLayersNetworkDiffrentOptimizer(nn.Module):
    
    def __init__(self, input_dim, fc1_dim, fc2_dim,fc3_dim , fc4_dim, n_action, lr,loss):
        super(MoreLayersNetworkDiffrentOptimizer, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(

              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, fc2_dim),
              nn.ReLU(),
              nn.Linear(fc2_dim, fc3_dim),
              nn.ReLU(),
              nn.Linear(fc3_dim, n_action),

        )

        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        
        self.loss = loss
    def forward(self, state):
        actions = self.network(state)
        return actions



class SimpleDiffrentLossFunction(nn.Module):
    
    def __init__(self, input_dim, fc1_dim, fc2_dim,fc3_dim , fc4_dim, n_action, lr, loss):
        super(SimpleDiffrentLossFunction, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(

              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, n_action),
              
      

        )
 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.loss = loss
        
    def forward(self, state):
        actions = self.network(state)
        return actions
    
class MoreLayerDiffrentLossFunction(nn.Module):
    
    def __init__(self, input_dim, fc1_dim, fc2_dim,fc3_dim , fc4_dim, n_action, lr , loss ):
        super(MoreLayerDiffrentLossFunction, self).__init__()
        self.lr = lr

        self.network = nn.Sequential(

              nn.Linear(input_dim, fc1_dim),
              nn.ReLU(),
              nn.Linear(fc1_dim, fc2_dim),
              nn.ReLU(),
              nn.Linear(fc2_dim, fc3_dim),
              nn.ReLU(),
              nn.Linear(fc3_dim, n_action),

        )
 

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        
        self.loss = loss 
        
    def forward(self, state):
        actions = self.network(state)
        return actions
    
class Qnetwork(nn.Module):
        
        def __init__(self, input_dim, fc1_dim, fc2_dim,fc3_dim , fc4_dim, n_action, lr ,loss):
            super(Qnetwork, self).__init__()
            self.lr = lr
            self.network = nn.Sequential(
                nn.Linear(input_dim, fc1_dim),
                nn.ReLU(),
                nn.Linear(fc1_dim, n_action),
            
            )
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.loss = loss
            
        def forward(self, state):
            actions = self.network(state)
            return actions


