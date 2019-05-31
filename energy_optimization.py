"""
optimize the energy function: find A which minimizes the energy \sum_((A - 10) ** 10)
"""
import torch
from torch.autograd import Variable

# - Your energy optimization problem is non-convex. If your initial estimation is too far away from local optima, your solution may not be optimal. Code from (Section 3, Q2) can be used to see how far you are from the original face.
# - Equation 4: p_{kid}_j should be p_{k}_j
# - Landmarks are a subset of vertices from the morphable model (indexes are defined by the annotation file provided), that's why you are inferring landmarks.

lr = 0.1
A = Variable(torch.ones(1, 10), requires_grad=True)
print('Initial A = ', A)
opt = torch.optim.Adam([A], lr=lr)

for i in range(100):
    opt.zero_grad()
    loss = torch.sum((A - 10) ** 10)
    loss.backward()
    opt.step()
print("Answer: A = ", A)
