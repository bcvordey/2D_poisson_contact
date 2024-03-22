####### 2D Poisson with new BC #######
# solve the Poisson equation -Delta u = f
#with Dirichlet boundary condition u = 0 on left|bottom|top
# Robin BC on right               -u_x = gamma(u-1)

from ngsolve import *
from netgen.geom2d import unit_square
import ngsolve.meshes as ngm

####### known term #####
# gamma = penalty parameter 
gamma = 10 ## increase to 1000 then 10000 then solution approaches signorini  
rhs = 2

n_elem = 4
mesh = ngm.MakeStructured2DMesh(quads=False, nx=n_elem, ny=n_elem)

# Non-homogeneous Dirichlet on top and bottom
fes = H1(mesh, order=2, dirichlet="left|top|bottom")
g = 0

#Putting g on the boundaries marked non-homogeneous Dirichlet
gfu = GridFunction(fes)
gfu.Set(g, BND)
#Draw(gfu)


#Bilinear form
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx + gamma*u*v*ds(definedon = 'right')).Assemble()

#Linear Form
f = LinearForm(rhs*v*dx + gamma*v*ds(definedon = 'right')).Assemble()


#solving
gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec 
Draw(gfu)
# Draw(-grad(gfu), mesh, "Flux")

#Error 
exact = x*x - 1/2 * y*y
print ("L2-error:", sqrt(Integrate((gfu-exact)*(gfu-exact), mesh)))


# Draw(gfu.components[0],deformation=True, settings={"camera": {"transformations": [{"type": "rotateX", "angle": -45}]}})