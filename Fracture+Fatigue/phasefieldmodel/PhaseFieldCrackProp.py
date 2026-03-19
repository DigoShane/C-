from dolfin import *
import numpy as np

# -------------------------
# Mesh
# -------------------------
mesh = RectangleMesh(Point(0, 0), Point(1.0, 1.0), 80, 80)

# -------------------------
# Function spaces
# -------------------------
Vu = VectorFunctionSpace(mesh, "CG", 1)
Vd = FunctionSpace(mesh, "CG", 1)

u = Function(Vu, name="Displacement")
phi = Function(Vd, name="Damage")
phi_old = Function(Vd)

v = TestFunction(Vu)
vphi = TestFunction(Vd)

# -------------------------
# Material parameters
# -------------------------
E = 210e9
nu = 0.3
Gc = 2.7
l0 = 0.02
eta = 1e-6

mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))

# -------------------------
# Kinematics
# -------------------------
def eps(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(eps(u))*Identity(2) + 2*mu*eps(u)

def psi_plus(u):
    eps_u = eps(u)
    tr_eps = tr(eps_u)
    
    # Macaulay bracket: <tr(ε)>₊
    tr_pos = (tr_eps + abs(tr_eps)) / 2.0
    
    # Volumetric part (tension only) + full deviatoric part
    return 0.5 * lmbda * tr_pos**2 + mu * inner(eps_u, eps_u)

# -------------------------
# Degradation
# -------------------------
g = (1 - phi)**2 + eta

# -------------------------
# History field
# -------------------------
Vh = FunctionSpace(mesh, "DG", 0)
H = Function(Vh)

def update_history():
    psi_vals = project(psi_plus(u), Vh)
    H.vector()[:] = np.maximum(H.vector().get_local(),
                               psi_vals.vector().get_local())

# -------------------------
# Boundary conditions
# -------------------------
def top(x, on_boundary):
    return near(x[1], 1.0) and on_boundary

def bottom(x, on_boundary):
    return near(x[1], 0.0) and on_boundary

def left(x, on_boundary):
    return near(x[0], 0.0) and on_boundary

U_top = Expression(("0.0", "t"), t=0.0, degree=1)
U_bottom = Expression(("0.0", "-t"), t=0.0, degree=1)

bc_top = DirichletBC(Vu, U_top, top)
bc_bottom = DirichletBC(Vu, U_bottom, bottom)
bc_left = DirichletBC(Vu.sub(0), Constant(0.0), left)

bcu = [bc_top, bc_bottom, bc_left]

# -------------------------
# Initial crack
# -------------------------
class InitialCrack(UserExpression):
    def eval(self, value, x):
        if abs(x[1]-0.5) < mesh.hmin() and x[0] < 0.3:
            value[0] = 1.0
        else:
            value[0] = 0.0
    def value_shape(self):
        return ()

phi.interpolate(InitialCrack(degree=1))
phi_old.assign(phi)

# Pre-load history to prevent damage from collapsing at step 1
H.vector()[:] = Gc / (2.0 * l0)

# -------------------------
# Degraded stress
# -------------------------
def sigma_degraded(u, phi):
    eps_u = eps(u)
    tr_eps = tr(eps_u)
    tr_pos = (tr_eps + abs(tr_eps)) / 2.0
    tr_neg = tr_eps - tr_pos

    sigma_pos = lmbda * tr_pos * Identity(2) + 2*mu*eps_u
    sigma_neg = lmbda * tr_neg * Identity(2)

    g_val = (1 - phi)**2 + eta
    return g_val * sigma_pos + sigma_neg

# -------------------------
# Linear elasticity problem
# -------------------------
du = TrialFunction(Vu)
a_u = inner(sigma_degraded(du, phi), eps(v)) * dx
L_u = dot(Constant((0.0, 0.0)), v) * dx

problem_u = LinearVariationalProblem(a_u, L_u, u, bcs=bcu)
solver_u = LinearVariationalSolver(problem_u)

# -------------------------
# Phase-field (nonlinear)
# -------------------------
F_phi = (
    (Gc/l0)*phi*vphi*dx
    + Gc*l0*dot(grad(phi), grad(vphi))*dx
    - 2*(1 - phi)*H*vphi*dx
)

J_phi = derivative(F_phi, phi)

problem_phi = NonlinearVariationalProblem(F_phi, phi, J=J_phi)
solver_phi = NonlinearVariationalSolver(problem_phi)

solver_phi.parameters["newton_solver"]["relative_tolerance"] = 1e-5
solver_phi.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
solver_phi.parameters["newton_solver"]["maximum_iterations"] = 25
solver_phi.parameters["newton_solver"]["relaxation_parameter"] = 0.6

# -------------------------
# Output
# -------------------------
xdmf_u = XDMFFile("mode1_displacement.xdmf")
xdmf_phi = XDMFFile("mode1_damage.xdmf")

for f in [xdmf_u, xdmf_phi]:
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True

# -------------------------
# Load stepping
# -------------------------
load_steps = 200
u_max = 0.02

for step in range(load_steps):

    load = (step+1)/load_steps * u_max
    U_top.t = load
    U_bottom.t = load

    # Staggered minimization
    for k in range(5):

        solver_u.solve()
        update_history()

        phi_old.assign(phi)
        solver_phi.solve()

        phi.vector()[:] = np.maximum(phi.vector().get_local(),
                                     phi_old.vector().get_local() - 1e-8)

    print(f"Step {step+1}, load={load:.5f}")

    xdmf_u.write(u, step)
    xdmf_phi.write(phi, step)

xdmf_u.close()
xdmf_phi.close()
