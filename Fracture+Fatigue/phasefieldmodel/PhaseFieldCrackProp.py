from dolfin import *
import numpy as np

# -------------------------
# Mesh (rectangular plate)
# -------------------------
mesh = RectangleMesh(Point(0, 0), Point(1.0, 1.0), 100, 100)

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

M = 1.0
dt = 1e-4

mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))

# -------------------------
# Kinematics
# -------------------------
def eps(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(eps(u))*Identity(2) + 2*mu*eps(u)

def psi(u):
    return 0.5*inner(sigma(u), eps(u))

# -------------------------
# Degradation function
# -------------------------
g = (1 - phi)**2 + eta

# -------------------------
# History field (irreversibility)
# -------------------------
H = Function(Vd)
H.vector()[:] = 0.0

def update_history():
    psi_proj = project(psi(u), Vd)
    H.vector()[:] = np.maximum(H.vector().get_local(),
                               psi_proj.vector().get_local())

# -------------------------
# Boundary conditions (Mode I)
# -------------------------
def top(x, on_boundary):
    return near(x[1], 1.0) and on_boundary

def bottom(x, on_boundary):
    return near(x[1], 0.0) and on_boundary

def left(x, on_boundary):
    return near(x[0], 0.0) and on_boundary

# symmetric opening displacement
U_top = Expression(("0.0", "t"), t=0.0, degree=1)
U_bottom = Expression(("0.0", "-t"), t=0.0, degree=1)

bc_top = DirichletBC(Vu, U_top, top)
bc_bottom = DirichletBC(Vu, U_bottom, bottom)

# fix one point horizontally to prevent rigid motion
bc_left = DirichletBC(Vu.sub(0), Constant(0.0), left)

bcu = [bc_top, bc_bottom, bc_left]

# -------------------------
# Initial crack (notch)
# -------------------------
class InitialCrack(UserExpression):
    def eval(self, value, x):
        if abs(x[1]-0.5) < 0.01 and x[0] < 0.3:
            value[0] = 1.0
        else:
            value[0] = 0.0
    def value_shape(self):
        return ()

phi.interpolate(InitialCrack(degree=1))
phi_old.assign(phi)

# -------------------------
# Displacement problem (linear)
# -------------------------
du = TrialFunction(Vu)
a_u = inner(g * sigma(du), eps(v)) * dx
L_u = Constant((0.0, 0.0))
L_u_form = dot(L_u, v) * dx

# -------------------------
# Phase-field evolution (nonlinear)
# -------------------------
F_phi = (
    (phi - phi_old)/dt * vphi * dx
    + M*(
        (Gc/l0)*phi*vphi*dx
        + Gc*l0*dot(grad(phi), grad(vphi))*dx
        - 2*(1 - phi)*H*vphi*dx
    )
)

J_phi = derivative(F_phi, phi)
problem_phi = NonlinearVariationalProblem(F_phi, phi, J=J_phi)
solver_phi = NonlinearVariationalSolver(problem_phi)

solver_phi.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
solver_phi.parameters["newton_solver"]["relative_tolerance"] = 1e-7
solver_phi.parameters["newton_solver"]["maximum_iterations"] = 20

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
load_steps = 100
time_steps = 10
u_max = 0.02

for step in range(load_steps):

    load = (step+1)/load_steps * u_max
    U_top.t = load
    U_bottom.t = load

    # update linear elasticity problem
    problem_u = LinearVariationalProblem(a_u, L_u_form, u, bcs=bcu)
    solver_u = LinearVariationalSolver(problem_u)

    for t in range(time_steps):

        for k in range(5):  # staggered iterations

            # Solve elasticity
            solver_u.solve()

            # Update history field
            update_history()

            # Solve phase-field
            solver_phi.solve()

            # Enforce irreversibility
            phi.vector()[:] = np.maximum(phi.vector().get_local(),
                                         phi_old.vector().get_local())

            phi_old.assign(phi)

    print(f"Step {step+1}/{load_steps}, load={load:.5f}")

    xdmf_u.write(u, step)
    xdmf_phi.write(phi, step)

xdmf_u.close()
xdmf_phi.close()
