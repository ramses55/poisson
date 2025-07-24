import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter

from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from matplotlib import cm



class model(nn.Module):
    """ A simple MLP

    Args:
        width: Number of neurons in each inner layers
        num_inner: Number of inner layers
    """

    def __init__(self,width=20,num_inner=1):
        super().__init__()
        self.first = nn.Linear(2,width)
        self.linears = nn.ModuleList([nn.Linear(width,width) for i in range(num_inner)])
        self.last = nn.Linear(width,1)

    def forward(self,x):
        res=self.first(x)
        for layer in self.linears:
            res = torch.tanh(layer(res))

        res = self.last(res)
        return res


class equation():
    """ Basic Poisson equation in rectangle with constant border conditions

    Args:
        con[1-4]: Constant value on corresponding side of rectangle
        xmin,xmax,ymin,ymax: Points defining a rectangle
    """

    def __init__(self,model,con1,con2,con3,con4,
                 xmin,ymin,xmax,ymax):
        self.pinn=model
        self.con1=con1
        self.con2=con2
        self.con3=con3
        self.con4=con4
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax


    def source(self,point):
        """ Source function in Poisson equation """

        x=point[:,0].reshape(-1,1)
        y=point[:,1].reshape(-1,1)
        return -2*np.pi**2*torch.sin(np.pi*x)*torch.sin(np.pi*y)


    def derivatives(self,point):
        """ Calculation of Laplas operator """

        point.requires_grad_(True)
        f=self.pinn(point)

        gr1=torch.ones_like(f)
        f_x=grad(f,point,grad_outputs=gr1,create_graph=True,retain_graph=True)[0]
        
        gr2=torch.ones_like(f_x)
        gr2[:,1]=0
        f_xx=grad(f_x,point,grad_outputs=gr2,create_graph=True,retain_graph=True)[0]
        u1=(f_xx[:,0]).reshape(-1,1)
        
        
        gr3=torch.ones_like(f_x)
        gr3[:,0]=0
        f_xx=grad(f_x,point,grad_outputs=gr3,create_graph=True,retain_graph=True)[0]
        u2=(f_xx[:,1]).reshape(-1,1)

        laplas=u1+u2

        return laplas

    def border_points(self,point_number):
        """ Generation of border points """

        xmin=self.xmin
        ymin=self.ymin
        xmax=self.xmax
        ymax=self.ymax



        x1=torch.linspace(xmin,xmax,point_number)
        y1=torch.linspace(ymin,ymin,point_number)
        b1=torch.stack((x1,y1),axis=-1).reshape(-1,2)
        
        
        x2=torch.linspace(xmin,xmax,point_number)
        y2=torch.linspace(ymax,ymax,point_number)
        b2=torch.stack((x2,y2),axis=-1).reshape(-1,2)
        
        
        x3=torch.linspace(xmin,xmin,point_number)
        y3=torch.linspace(ymin,ymax,point_number)
        b3=torch.stack((x3,y3),axis=-1).reshape(-1,2)
        
        
        
        x4=torch.linspace(xmax,xmax,point_number)
        y4=torch.linspace(ymin,ymax,point_number)
        b4=torch.stack((x4,y4),axis=-1).reshape(-1,2)
        
        b=list((b1,b2,b3,b4))
    
        return b


    def loss(self,point):

        """ Loss function
            Returns:
                tuple of different losses:
                    l --- loss
                    lf --- loss corresponding to function
                    lb[1-4] --- loss corresponding to each of 4 borders
        """

        laplas=self.derivatives(point)
        s=self.source(point)

        b=self.border_points(100)

        lf=torch.mean((laplas-s)**2)
        lb1=torch.mean((self.pinn(b[0])-self.con1)**2)
        lb2=torch.mean((self.pinn(b[1])-self.con2)**2)
        lb3=torch.mean((self.pinn(b[2])-self.con3)**2)
        lb4=torch.mean((self.pinn(b[3])-self.con4)**2)


        l=lf+lb1+lb2+lb3+lb4


        return (l,lf,lb1,lb2,lb3,lb4)



def train_loop(equation,mesh,a_epochs,l_epochs,writer):
    """ Training consists of two steps:
            1) optimization using Adam
            2) optimization using LBFGS
        Args:
            equation --- member of equation class
            mesh --- collocation points in rectangle
            a_epochs --- number of epochs for Adam 
            l_epochs --- number of epochs for LBFGS
            writer --- writer for Tensor Board
        Returns:
            tuple: trained PINN and loss
    """
    optimizer=torch.optim.Adam(equation.pinn.parameters(),lr=1e-3)
    for epoch in range(a_epochs):
        optimizer.zero_grad()
        loss,lf,lb1,lb2,lb3,lb4 = equation.loss(mesh)

        writer.add_scalar('bc1',lb1,epoch)
        writer.add_scalar('bc2',lb2,epoch)
        writer.add_scalar('bc3',lb3,epoch)
        writer.add_scalar('bc4',lb4,epoch)
        writer.add_scalar('lf',lf,epoch)
        writer.add_scalar('l',loss,epoch)

        loss.backward()
        optimizer.step()
    
        if epoch % (a_epochs / 10) == 0:
            print(f"{epoch}:{loss:e}")

    print("LFBGS")
    lbfgs = torch.optim.LBFGS(equation.pinn.parameters(), lr=1,max_iter=1000)
    
    def closure():
            lbfgs.zero_grad()
            loss,lf,lb1,lb2,lb3,lb4 = equation.loss(mesh)
            loss.backward()
            return loss
    
    
    for epoch in range(l_epochs):
        loss=lbfgs.step(closure)
    print(f"Final loss:{loss:e}")
    return equation.pinn, loss






def plot(model,xmin,ymin,xmax,ymax,point_number):
    """ Plots graph and saves to 'graph.png' 
        Args:
            model --- trained PINN returned by train_loop
            [xy][min,max] --- rectangle 
            point_number --- number of points to plot

    """
    
    x=torch.linspace(xmin,xmax,point_number)
    y=torch.linspace(ymin,ymax,point_number)
    
    X,Y=torch.meshgrid((x,y),indexing='ij')
    
    x=X.reshape(-1,1)
    y=Y.reshape(-1,1)
    
    
    points=torch.stack((x,y),axis=-1).reshape(-1,2)
    
    X=X.detach().numpy()
    Y=Y.detach().numpy()
    Z=model(points).reshape(X.shape).detach().numpy()
    
    
    
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10) 
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, 20, cmap=cm.coolwarm)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.savefig("graph.png")
    plt.show()

def l2re(model,xmin,ymin,xmax,ymax,point_number):
    """ Caclculates L2 relative error """

    x=torch.linspace(xmin,xmax,point_number)
    y=torch.linspace(ymin,ymax,point_number)
    
    X,Y=torch.meshgrid((x,y),indexing='ij')
    
    x=X.reshape(-1,1)
    y=Y.reshape(-1,1)
    
    
    points=torch.stack((x,y),axis=-1).reshape(-1,2)
    true=torch.sin(np.pi*x)*torch.sin(np.pi*y)
    pred=model(points)

    l2re=torch.sqrt(torch.sum(true-pred)**2/torch.sum(true))
    return l2re






