import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# -------------------------
# Mesh
# -------------------------
domain = mesh.create_unit_square(MPI.COMM_WORLD, 100, 100)
dx = ufl.dx(domain)

# -------------------------
# Function spaces
# -------------------------
V_u = fem.functionspace(domain, ("CG", 1, (2,)))
V_phi = fem.functionspace(domain, ("CG", 1))

u = fem.Function(V_u)
phi = fem.Function(V_phi)
phi_old = fem.Function(V_phi)

v = ufl.TestFunction(V_u)
vphi = ufl.TestFunction(V_phi)

H = fem.Function(V_phi)
H.x.array[:] = 0.0


# -------------------------
# Material parameters
# -------------------------
E = 210e9
nu = 0.3
Gc = 1.0
l = 0.02
eta = 1e-6
M = 1.0
dt = 1e-4

mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda*ufl.tr(eps(u))*ufl.Identity(2) + 2*mu*eps(u)

def psi(u):
    return 0.5*ufl.inner(sigma(u), eps(u))

# -------------------------
# Degradation function
# -------------------------
g = (1 - phi)**2 + eta

# -------------------------
# Variational forms
# -------------------------
F_u = ufl.inner(g*sigma(u), eps(v)) * dx

F_phi = ((phi - phi_old)/dt)*vphi*dx + M*((Gc/l)*phi*vphi*dx + Gc*l*ufl.dot(ufl.grad(phi), ufl.grad(vphi))*dx - 2*(1-phi)*H*vphi*dx)

# -------------------------
# Boundary conditions
# -------------------------
fdim = domain.topology.dim - 1

def top(x):
    return np.isclose(x[1], 1.0)

def bottom(x):
    return np.isclose(x[1], 0.0)

top_facets = mesh.locate_entities_boundary(domain, fdim, top)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)

# Bottom fixed
dofs_bottom = fem.locate_dofs_topological(V_u, fdim, bottom_facets)
bc_bottom = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), dofs_bottom, V_u)

# Top: only vertical displacement
dofs_top_y = fem.locate_dofs_topological(V_u.sub(1), fdim, top_facets)

# -------------------------
# Initial crack
# -------------------------
def crack(x):
    values = np.zeros(x.shape[1])
    mask = (np.abs(x[1]-0.5) < 0.01) & (x[0] < 0.3)
    values[mask] = 1.0
    return values

phi.interpolate(crack)
phi_old.x.array[:] = phi.x.array

# -------------------------
# Boundary markers (for force)
# -------------------------
boundaries = mesh.meshtags(domain, fdim, top_facets,
                           np.full(len(top_facets), 1, dtype=np.int32))

ds = ufl.Measure("ds", domain=domain, subdomain_data=boundaries)

# -------------------------
# Solvers
# -------------------------
du = ufl.TrialFunction(V_u)

a_u = ufl.inner(g * sigma(du), eps(v)) * dx

zero = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
L_u = ufl.dot(zero, v) * dx

problem_u = fem.petsc.LinearProblem(a_u, L_u, bcs=[bc_bottom],
                                   petsc_options={"ksp_type": "preonly",
                                                  "pc_type": "lu"})

problem_phi = NonlinearProblem(F_phi, phi)
solver_phi = NewtonSolver(MPI.COMM_WORLD, problem_phi)

# -------------------------
# Load stepping
# -------------------------
load_steps = 200
time_steps = 20
u_max = 0.02

file = io.XDMFFile(domain.comm, "phi.xdmf", "w")
file.write_mesh(domain)
time = 0.0

for step in range(load_steps):

    load = (step+1)/load_steps * u_max

    u_top = fem.Constant(domain, PETSc.ScalarType(load))
    bc_top = fem.dirichletbc(u_top, dofs_top_y, V_u.sub(1))

    problem_u = fem.petsc.LinearProblem(a_u, L_u, bcs=[bc_bottom, bc_top], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    for t in range(time_steps):
      for k in range(5):   # staggered iterations

        # Solve displacement
        u = problem_u.solve()

        psi_expr = fem.Expression(psi(u), V_phi.element.interpolation_points())
        psi_vals = fem.Function(V_phi)
        psi_vals.interpolate(psi_expr)

        H.x.array[:] = np.maximum(H.x.array, psi_vals.x.array)

        # Solve phase-field evolution
        solver_phi.solve(phi)

        # Irreversibility (no healing)
        phi.x.array[:] = np.maximum(phi.x.array, phi_old.x.array)

        # Update history
        phi_old.x.array[:] = phi.x.array
    
    #print step:
    if MPI.COMM_WORLD.rank == 0:
     print(f"\n--- Load step {step+1}/{load_steps}, load = {load:.6f} ---")

    n = ufl.FacetNormal(domain)
    traction = ufl.dot(g*sigma(u), n)
    force = fem.assemble_scalar(fem.form(traction[1]*ds(1)))

    if MPI.COMM_WORLD.rank == 0:
     print(f"Reaction force = {force:.6e}")

    file.write_function(phi, time)
    time += 1.0

file.close()
