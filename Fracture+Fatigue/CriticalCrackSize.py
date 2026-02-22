'''
Problem Formulation Fatigue Crack Growth and Remaining Life Assessment
of a Pressurized Cylindrical Vessel

1.  Engineering Context

We consider a thin-walled cylindrical pressure vessel subjected to
cyclic internal pressure. The goal is to determine:

1)  The critical crack size at which fracture occurs.
2)  The number of cycles to failure from an initial detectable crack.
3)  The remaining life as a function of crack size.

This represents a classic pressure vessel / pipeline integrity problem
aligned with fracture-based fitness-for-service (FFS) methodology.

2.  Geometry and Material Properties

Cylinder parameters: R = mean radius t = wall thickness (t << R) L =
length

Material properties: E = Young’s modulus, nu = Poisson’s ratio, K_IC =
Mode I fracture toughness C, m = Paris law constants

3.  Loading

The vessel experiences cyclic internal pressure:

P(t) = P_max * sin(omega t)

Under the thin-wall assumption, the dominant stress is hoop stress:

sigma_theta = (P * R) / t

Maximum hoop stress:

sigma_max = (P_max * R) / t

Stress range:

Delta_sigma = sigma_max - sigma_min

For fully reversed loading:

Delta_sigma = (2 * P_max * R) / t

4.  Crack Geometry

Assume a longitudinal surface crack of depth a, oriented parallel to the
cylinder axis. Hoop stress produces Mode I opening.

Stress intensity factor:

K_I(a) = Y(a/t) * sigma_theta * sqrt(pi * a)

Y(a/t) = geometry correction factor (approx 1.1–1.2 for shallow surface
cracks)

5.  Fracture Criterion

Fracture occurs when:

K_I(a_c) = K_IC

Critical crack size:

a_c = (1/pi) * (K_IC / (Y * sigma_max))^2

Assumptions: - Linear elastic fracture mechanics (LEFM) - Small-scale
yielding

6.  Fatigue Crack Growth Law

Paris law:

da/dN = C * (Delta_K)^m

Stress intensity range:

Delta_K = Y * Delta_sigma * sqrt(pi * a)

Substituting:

da/dN = C * (Y * Delta_sigma * sqrt(pi * a))^m

7.  Remaining Life Calculation

Total cycles to failure:

N_f = integral from a0 to a_c of da / [ C * (Y * Delta_sigma * sqrt(pi *
a))^m ]

For m ≠ 2, closed-form solution:

N_f = 2 / [ C * (Y * Delta_sigma)^m * pi^(m/2) * (2 - m) ] * [ a_c^((2 -
m)/2) - a0^((2 - m)/2) ]

8.  Engineering Outputs

-   Critical crack size (a_c)
-   Cycles to failure (N_f)
-   Crack growth curve a(N)
-   Remaining life vs crack size
-   Inspection interval recommendations

Engineering Significance:

This formulation mirrors fracture-mechanics-based fitness-for-service
methodology used in industrial integrity assessments. It provides
quantitative basis for remaining life estimation, inspection scheduling,
and defensible structural analysis.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# 1. Input Parameters
# ============================================================

# Geometry
R = 0.5            # m (mean radius)
t = 0.01           # m (wall thickness)

# Loading
P_max = 5e6        # Pa (5 MPa)
P_min = 0          # Pa

# Material properties
K_IC = 80e6        # Pa*sqrt(m)  (80 MPa√m)
C = 1e-11          # Paris constant (typical steel)
m_paris = 3.0      # Paris exponent
Y = 1.12           # geometry factor

# Initial crack size
a0 = 1e-3          # 1 mm

# ============================================================
# 2. Derived Quantities
# ============================================================

sigma_max = (P_max * R) / t
sigma_min = (P_min * R) / t
delta_sigma = sigma_max - sigma_min

# ============================================================
# 3. Critical Crack Size
# ============================================================

a_critical = (1/np.pi) * (K_IC / (Y * sigma_max))**2

print("Critical crack size (m):", a_critical)

# ============================================================
# 4. Closed-Form Fatigue Life
# ============================================================

if m_paris != 2:
    N_closed = (
        2 /
        (C * (Y * delta_sigma)**m_paris * np.pi**(m_paris/2) * (2 - m_paris))
    ) * (
        a_critical**((2 - m_paris)/2) - a0**((2 - m_paris)/2)
    )
else:
    raise ValueError("Closed-form not valid for m = 2")

print("Fatigue life (closed-form cycles):", N_closed)

# ============================================================
# 5. Numerical Crack Growth Integration
# ============================================================

def paris_law(N, a):
    delta_K = Y * delta_sigma * np.sqrt(np.pi * a)
    return C * delta_K**m_paris

sol = solve_ivp(
    paris_law,
    t_span=(0, N_closed),
    y0=[a0],
    max_step=N_closed/1000
)

N_values = sol.t
a_values = sol.y[0]

# ============================================================
# 6. Plot Results
# ============================================================

plt.figure()
plt.plot(N_values, a_values)
plt.xlabel("Cycles")
plt.ylabel("Crack depth (m)")
plt.title("Fatigue Crack Growth")
plt.grid(True)
plt.show()

# ============================================================
# 7. Log-Log Crack Growth Rate Plot
# ============================================================

a_test = np.linspace(a0, a_critical, 200)
delta_K_test = Y * delta_sigma * np.sqrt(np.pi * a_test)
da_dN_test = C * delta_K_test**m_paris

plt.figure()
plt.loglog(delta_K_test, da_dN_test)
plt.xlabel("Delta K (Pa sqrt(m))")
plt.ylabel("da/dN (m/cycle)")
plt.title("Paris Law")
plt.grid(True, which="both")
plt.show()
