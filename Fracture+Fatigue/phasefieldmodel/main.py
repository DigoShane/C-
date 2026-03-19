from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pygmsh

##Meshing
#N = 100
#lx = float(1) 
#ly = float(1) 
#geom = pygmsh.opencascade.Geometry()
## Rectangle: from (0,0) to (1,1)
#rect = geom.add_rectangle([0.0, 0.0, 0.0], lx, ly)
## Circular hole at center with radius 0.2
#circ = geom.add_disk([0.5, 0.5, 0.0], 0.2)
#
## Boolean operation: rectangle minus circle
#final_domain = geom.boolean_difference([rect], [circ])
#
## GENERATE MESH
#mesh = pygmsh.generate_mesh(geom, dim=2, geo_filename="rect_with_hole.geo")


#--------------------------------------------------------------------------
import gmsh
import meshio

# -------------------------
# Mesh (SENT specimen)
# -------------------------
L = 1.0
H = 1.0
nx, ny = 300, 300

mesh = RectangleMesh(Point(0, 0), Point(L, H), nx, ny)

#Function Spaces
Vu = VectorFunctionSpace(mesh, "CG", 1)   # displacement
Vd = FunctionSpace(mesh,       "CG", 1)   # damage / phase-field
Vs = TensorFunctionSpace(mesh, "DG", 0)   # stress (for output only)
 
u    = Function(Vu, name="Displacement")
d    = Function(Vd, name="Damage")
d_old = Function(Vd, name="Damage_old")   # previous iterate (stagger convergence)


# Material properties
E, nu = 200, 0.2
lmbda  = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
mu     = Constant(E / (2 * (1 + nu)))
kappa  = Constant(lmbda + 2.0/3.0 * mu)   # bulk modulus

kres  = Constant(1e-6)          # residual stiffness
Gc    = Constant(2.7)           # critical energy release rate
l0    = Constant(0.005)          # phase-field length scale

# -------------------------
# Initial crack (Miehe notch)
# -------------------------
class InitialCrack(UserExpression):
    def eval(self, values, x):
        if x[0] <= 0.3 and abs(x[1] - 0.5) < float(l0):
            values[0] = 1.0
        else:
            values[0] = 0.0

    def value_shape(self):
        return () 

d.interpolate(InitialCrack(degree=1))
d_old.assign(d)


# -------------------------
# Boundary conditions (Miehe SENT)
# -------------------------

def bottom(x, on_boundary):
    return near(x[1], 0.0) and on_boundary

def top(x, on_boundary):
    return near(x[1], H) and on_boundary

def left(x, on_boundary):
    return near(x[0], 0.0) and on_boundary

Uimp = Expression("t", t=0.0, degree=1)

def left_bottom_point(x, on_boundary):
    return near(x[0], 0.0) and near(x[1], 0.0)


bcu = [
    DirichletBC(Vu.sub(1), Constant(0.0), bottom),  # uy = 0
    DirichletBC(Vu.sub(1), Uimp, top),              # uy = t
    DirichletBC(Vu.sub(0), Constant(0.0),
                left_bottom_point,
                method="pointwise")                # ux = 0 at one point
]

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H) and on_boundary

TopBoundary().mark(boundaries, 1)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
n  = FacetNormal(mesh)

# Virtual field for reaction force recovery on top boundary
v_reaction = Function(Vu)
bcu[1].apply(v_reaction.vector())   # uy=1 on top, 0 elsewhere


#Kinematics — Volumetric/Deviatoric Split
def eps(v):
    """Symmetric small strain tensor."""
    return sym(grad(v))
 
def tr_pos(A):
    """Positive part of the trace: <tr(A)>+ = max(tr(A), 0)."""
    return (tr(A) + abs(tr(A))) / 2.0
 
def tr_neg(A):
    """Negative part of the trace: <tr(A)>- = min(tr(A), 0)."""
    return (tr(A) - abs(tr(A))) / 2.0
 
def psi_plus(v):
    """
    Tensile (positive) strain energy density — drives crack growth.
    W+ = kappa/2 * <tr(eps)>+^2  +  mu * dev(eps):dev(eps)
    Matches Ψ₊ in the MFront behaviour.
    """
    e    = eps(v)
    e_dev = e - (1.0/3.0) * tr(e) * Identity(2)   # deviatoric strain (2D)
    return (kappa / 2.0) * tr_pos(e)**2 + mu * inner(e_dev, e_dev)
 
def psi_minus(v):
    """
    Compressive (negative) strain energy density — does NOT drive cracks.
    W- = kappa/2 * <tr(eps)>-^2
    """
    return (kappa / 2.0) * tr_neg(eps(v))**2
 
def degradation(phi):
    """
    Quadratic degradation function  g(d) = (1-d)^2 + kres
    Matches gᵈ in the MFront behaviour.
    """
    return (1.0 - phi)**2 + kres
 
# History Field H
# (translates H = max(H, Ψ₊) in the MFront integrator)
# H is a DG0 field stored at quadrature/cell level.
# We use a DG0 projection for simplicity.
Vh = FunctionSpace(mesh, "DG", 0)
H  = Function(Vh, name="HistoryFunction")
H.vector()[:] = 0.0
 
def update_history():
    """H_new = max(H_old, W+(u)) — enforces crack irreversibility."""
    W_plus = project(psi_plus(u), Vh, solver_type="cg", preconditioner_type="ilu")
    H.vector()[:] = np.maximum(H.vector().get_local(),
                               W_plus.vector().get_local())
 
#Variational Forms
 
# ---- Displacement problem ----
# Degraded total energy:
#   Pi(u) = integral[ g(d)*W+(u)  +  W-(u) ] dV
# Weak form (Gateaux derivative w.r.t. u):
#   a_u(u, v) = L_u(v)
 
du = TrialFunction(Vu)
vu = TestFunction(Vu)
 
def sigma_degraded(v, phi):
    """
    Degraded Cauchy stress — translates the σ expression in MFront:
      σ = κ*(g(d)*<tr>+  +  <tr>-)* I  +  2μ*g(d)*dev(ε)
    """
    e = eps(v)
    e_vol = tr(e)
    e_vol_pos = (e_vol + abs(e_vol)) / 2.0
    e_vol_neg = (e_vol - abs(e_vol)) / 2.0
    e_dev = e - (1.0/3.0) * e_vol * Identity(2)

    sig_vol_pos = kappa * e_vol_pos * Identity(2)
    sig_vol_neg = kappa * e_vol_neg * Identity(2)
    sig_dev     = 2.0 * mu * e_dev

    return degradation(phi) * (sig_vol_pos + sig_dev) + sig_vol_neg

# Bilinear and linear forms for u (linear in du for fixed d)
F_u = inner(sigma_degraded(u, d), eps(vu)) * dx

# Nonlinear solver for displacement
du = TrialFunction(Vu)
J_u = derivative(F_u, u, du)

problem_u = NonlinearVariationalProblem(F_u, u, bcs=bcu, J=J_u)
solver_u  = NonlinearVariationalSolver(problem_u)

# Typical parameters (adjust as needed)
prm = solver_u.parameters["newton_solver"]

prm["absolute_tolerance"] = 1e-8
prm["relative_tolerance"] = 1e-7
prm["maximum_iterations"] = 25
prm["linear_solver"]      = "mumps"
prm["report"]             = True
prm["error_on_nonconvergence"] = False

dd  = TrialFunction(Vd)
q   = TestFunction(Vd)
 
def build_damage_forms():
    """Rebuild each stagger iteration because H changes."""
    a = ( (Gc/l0 + 2.0*H) * dd * q
        + Gc * l0 * dot(grad(dd), grad(q)) ) * dx
    L = 2.0 * H * q * dx
    return a, L
 
def solve_displacement():
    solver_u.solve() 

def solve_damage():
    a_d, L_d = build_damage_forms()
    solve(a_d == L_d, d, solver_parameters={"linear_solver": "lu"})          # or "mumps"
    d.vector()[:] = np.maximum(d.vector().get_local(),
                           d_old.vector().get_local())
    d.vector()[:] = np.clip(d.vector().get_local(), 0.0, 1.0)
 
#Energy functionals
def stored_energy():
    """Elastic strain energy: integral[ g(d)*W+(u) + W-(u) ] dV"""
    return assemble(
        (degradation(d) * psi_plus(u) + psi_minus(u)) * dx
    )
 
def dissipated_energy():
    """Fracture surface energy: integral[ Gc/(2l0)*d^2 + Gc*l0/2*|grad d|^2 ] dV"""
    return assemble(
        (Gc/(2*l0) * d**2 + Gc*l0/2 * dot(grad(d), grad(d))) * dx
    )

def sample_sigma_along_line(sigma_proj, y=0.5, npts=200):
    xs = np.linspace(0.31, 0.8, npts)
    sig_xx = []

    for x in xs:
        val = sigma_proj(Point(x, y))[0]
        sig_xx.append(val)

    return xs, np.array(sig_xx)


#ParaView Output
xdmf_u = XDMFFile("phase_field_no_mfront_displacement.xdmf")
xdmf_d = XDMFFile("phase_field_no_mfront_damage.xdmf")
 
for f in [xdmf_u, xdmf_d]:
    f.parameters["flush_output"]         = True
    f.parameters["functions_share_mesh"] = True
 
#Load-Stepping Loop
tol, Nitermax = 1e-2, 100

loading = np.linspace(0, 0.1, 800)
N_steps = loading.shape[0]
results = np.zeros((N_steps, 3))   # [force, elastic energy, fracture energy]
 
for i, t in enumerate(loading):
    print("Time step: {}  (u_imp = {:.4f})".format(i+1, t))
    Uimp.t = t
 
    # ---- Alternate minimization ----
    res = 1.0
    j   = 1
    while res > tol and j < Nitermax:
        # Step A: solve mechanics with current d
        solve_displacement()
 
        # Update history field H = max(H, W+(u))
        update_history()
 
        # Step B: solve damage with current u and H
        d_old.assign(d)
        solve_damage()
 
        # Convergence: max pointwise damage increment
        res = np.linalg.norm(d.vector().get_local() - d_old.vector().get_local(), ord=np.inf)
        print("   Iteration {:3d}:  max(Δd) = {:.2e}".format(j, res))
        j += 1

    if i == int(0.5 * N_steps):   # mid-load step
        sigma_proj = project(sigma_degraded(u, d), Vs)
        xs, sig = sample_sigma_along_line(sigma_proj)
 
    # ---- Post-processing ----
    sigma_proj = project(sigma_degraded(u, d), Vs)
    traction = dot(sigma_degraded(u, d), n)
    reaction = assemble(dot(traction, as_vector((0,1))) * ds(1))
    results[i, 0] = reaction
    results[i, 1] = stored_energy()
    results[i, 2] = dissipated_energy()
 
    xdmf_u.write(u, t)
    xdmf_d.write(d, t)
 
    clear_output(wait=True)
    plt.figure()
    p = plot(d, vmin=0, vmax=1)
    plt.colorbar(p)
    plt.title("Damage  t={:.4f}".format(t))
    plt.savefig("./results/phase_field_{:04d}.png".format(i), dpi=400)
    plt.close()
 
xdmf_u.close()
xdmf_d.close()



r = xs - 0.3  # distance from crack tip

plt.loglog(r, np.abs(sig), label="FEM")
plt.loglog(r, r**(-0.5), '--', label="r^{-1/2}")
plt.legend()
plt.xlabel("r")
plt.ylabel("|σ_xx|")
plt.title("Check ~ r^{-1/2} scaling")
plt.show()


#Summary Plots
plt.figure()
plt.plot(loading, results[:, 0], "-o")
plt.xlabel("Imposed displacement")
plt.ylabel("Vertical force")
plt.title("Load-displacement curve")
plt.show()
 
plt.figure()
plt.plot(loading, results[:, 1], label="elastic energy")
plt.plot(loading, results[:, 2], label="fracture energy")
plt.plot(loading, results[:, 1] + results[:, 2], label="total energy")
plt.xlabel("Imposed displacement")
plt.ylabel("Energies")
plt.legend()
plt.title("Energy evolution")
plt.show()
