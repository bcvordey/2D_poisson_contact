####### Neumann on the right #######
# solve the Poisson equation -Delta u = f
#with Dirichlet boundary condition u = 0 on left|bottom|top
# Neumann BC on right

from ngsolve import *
from netgen.geom2d import unit_square
import ngsolve.meshes as ngm

n_elem = 16
mesh = ngm.MakeStructured2DMesh(quads=False, nx=n_elem, ny=n_elem)

# Non-homogeneous Dirichlet on top and bottom
fes = H1(mesh, order=2, dirichlet="left|top|bottom")
# g = 1/2*x

#Putting g on the boundaries marked non-homogeneous Dirichlet
gfu = GridFunction(fes)
# gfu.Set(g, BND)
#Draw(gfu)


#Bilinear form
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble();

#Linear Form
f = LinearForm(1*v*dx).Assemble()
r = f.vec - a.mat * gfu.vec

#solving
gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * r
Draw(gfu)
Draw(-grad(gfu), mesh, "Flux")

#Error 
exact = x*x - 1/2 * y*y
print ("L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))


# Draw(gfu.components[0],deformation=True, settings={"camera": {"transformations": [{"type": "rotateX", "angle": -45}]}})