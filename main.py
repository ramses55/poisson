from functions import model, equation, train_loop, plot, l2re

import torch

from torch.utils.tensorboard import SummaryWriter

points_number = 1000
x = torch.rand(points_number,1)
y = torch.rand(points_number,1)



mesh = torch.stack((x,y),axis=-1).reshape(-1,2)

pinn = model(width=20,num_inner=1)
pinn.train(True)


writer = SummaryWriter('./runs/1')

equation = equation(pinn,con1=0,con2=0,con3=0,con4=0,
                  xmin=0,ymin=0,xmax=1,ymax=1)


pinn, loss = train_loop(equation=equation,mesh=mesh,a_epochs=1000,
                        l_epochs=100,writer=writer)

torch.save(pinn.state_dict(), f'model-{loss}.pth')

plot(pinn,xmin=0,ymin=0,xmax=1,ymax=1,point_number=100)

metric=l2re(pinn,xmin=0,ymin=0,xmax=1,ymax=1,point_number=100)
print(f"L2RE metrtic:{metric:e}")
