import torch.nn as nn
import torch
A = nn.Linear(10,10)
B = nn.Linear(10,10)
C = nn.Linear(10,10)
x = torch.rand(1,10, requires_grad=True)
yA = A(x).
yB = B(yA)
# with torch.no_grad():
#     yB = B(yA)
yC = C(yB)
yC.mean().backward()
print("end")