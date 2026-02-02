#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

namespace gl
{
  using namespace dealii;

  template <int dim>
  class PsiBoundary : public Function<dim>
  {
  public:
    PsiBoundary() : Function<dim>(2) {} // 2 components: u and v

    double value(const Point<dim> &, const unsigned int component) const override
    {
      // psi = 1 => u=1, v=0 on boundary
      return (component == 0) ? 1.0 : 0.0;
    }
  };

  template <int dim>
  class GLProblem
  {
  public:
    explicit GLProblem(const unsigned int degree);

    void set_kappa(const double k) { kappa_ = k; }
    void set_H(const double H) { H_ = H; }

    void run();

  private:
    void make_grid();
    void setup_system();

    void assemble_system();         // assemble Jacobian + RHS = -Residual
    bool solve_newton_step();       // solve linearized system
    bool newton_solve();            // Newton loop

    void output_results(const unsigned int it) const;

    // Return A and |A|^2 at point x (symmetric gauge for uniform B=H)
    Tensor<1, dim> A_field(const Point<dim> &x) const;
    double         A_sq(const Point<dim> &x) const;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;          // 2 scalar components: u and v
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;        // current Newton iterate (u,v)
    Vector<double> newton_update;   // delta
    Vector<double> system_rhs;      // -Residual

    QGauss<dim> quadrature;
    QGauss<dim-1> face_quadrature;

    // Parameters
    double kappa_ = 2.0;
    double H_     = 0.0;  // applied field strength

    // Newton controls
    unsigned int max_newton_iters_ = 25;
    double newton_tol_rel_ = 1e-10;
    double newton_tol_abs_ = 1e-12;

    unsigned int output_every_ = 1;
  };
} // namespace gl

