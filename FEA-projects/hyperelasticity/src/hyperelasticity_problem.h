#pragma once

// Core deal.II includes for quadrature, tensor-valued boundary loads, FEM types
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

// Linear algebra containers and constraints
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

// Mesh and DoF management
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

// Finite elements
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

namespace dealii_hyper
{
  using namespace dealii;

  // --------------------------------------------------------------------------
  // RightTraction: Neumann boundary condition (traction) on the right boundary.
  //
  // We apply a constant traction vector t = (T, 0) in 2D.
  // In the solver, T will be scaled by a load factor alpha in [0,1].
  // --------------------------------------------------------------------------
  template <int dim>
  class RightTraction : public TensorFunction<1, dim>
  {
  public:
    explicit RightTraction(const double T) : TensorFunction<1, dim>(), T_(T) {}

    Tensor<1, dim> value(const Point<dim> &) const override
    {
      Tensor<1, dim> t;
      t[0] = T_;
      for (unsigned int i = 1; i < dim; ++i)
        t[i] = 0.0;
      return t;
    }

  private:
    double T_;
  };


  // --------------------------------------------------------------------------
  // HyperElasticityProblem: 2D nonlinear FEA solver (compressible Neo-Hookean).
  //
  // Unknown field:
  //   displacement u(X) in R^dim, represented with a vector-valued FE system:
  //     FESystem<dim>(FE_Q<dim>(degree), dim)
  //
  // Nonlinear solve:
  //   For each load step alpha:
  //     Newton iterations to solve R(u)=0.
  //
  // We store:
  //   - solution: current displacement iterate
  //   - system_rhs: "-R(u)" (so solving K * du = system_rhs updates u)
  //   - system_matrix: Jacobian (tangent stiffness)
  // --------------------------------------------------------------------------
  template <int dim>
  class HyperElasticityProblem
  {
  public:
    explicit HyperElasticityProblem(const unsigned int degree);
    void run();

  private:
    // Build a rectangular mesh and tag boundaries.
    void make_grid();
    void set_boundary_ids();

    // Distribute DoFs, create constraints, allocate matrices/vectors.
    void setup_system();

    // Assemble nonlinear residual and Jacobian for a given load factor.
    void assemble_system(const double load_factor);

    // Solve one Newton linear system: K * delta_u = rhs
    bool solve_newton_step();

    // Full Newton loop for a given load factor.
    bool newton_solve(const double load_factor);

    // Output to VTU per load step.
    void output_results(const unsigned int step) const;

    // Material model helper:
    // Given F (deformation gradient), compute:
    //   P    = First Piola stress
    //   dPdF = tangent (Jacobian contribution)
    void compute_P_and_tangent(const Tensor<2, dim> &F,
                               Tensor<2, dim>       &P,
                               Tensor<4, dim>       &dPdF) const;

    // ---------------- FEM core objects ----------------
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;          // vector-valued displacement space
    DoFHandler<dim>    dof_handler; // DoF numbering + mapping

    // Constraints for Dirichlet boundary (and other constraints if added later)
    AffineConstraints<double> constraints;

    // Sparse linear system objects used in each Newton iteration
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;  // Jacobian / tangent stiffness

    Vector<double> solution;       // current u
    Vector<double> newton_update;  // delta_u
    Vector<double> system_rhs;     // -R(u)

    // Quadrature rules for cell and face integrals
    QGauss<dim>     quadrature;
    QGauss<dim - 1> face_quadrature;

    // ---------------- Problem parameters ----------------
    // Geometry: rectangle [0,L] x [0,H]
    double L = 1.0;
    double H = 0.2;

    // Material (Lamé constants)
    double lambda = 1.0;
    double mu     = 1.0;

    // Reference traction magnitude
    double traction_T = 1e-2;

    // Newton controls
    unsigned int max_newton_iters = 20;
    double       newton_tol_rel   = 1e-10;
    double       newton_tol_abs   = 1e-12;
  };

} // namespace dealii_hyper

