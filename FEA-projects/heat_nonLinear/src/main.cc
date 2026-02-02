#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
class NonlinearHeat
{
public:
  explicit NonlinearHeat(const unsigned int degree);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_MK();
  void assemble_residual_and_jacobian();
  bool solve_newton();
  void output(const unsigned int step) const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity;
  SparseMatrix<double> M, K;      // mass and stiffness
  SparseMatrix<double> J;         // Jacobian
  Vector<double>       R;         // -Residual stored in RHS form
  Vector<double>       u, u_old;  // current and previous timestep
  Vector<double>       du;        // Newton update

  QGauss<dim> quadrature;

  double kappa = 1.0;
  double dt    = 1e-2;
  double T     = 0.05;

  unsigned int max_newton = 20;
  double tol_rel = 1e-10;
  double tol_abs = 1e-12;
};

template <int dim>
NonlinearHeat<dim>::NonlinearHeat(const unsigned int degree)
  : fe(degree), dof_handler(triangulation), quadrature(degree + 1)
{}

template <int dim>
void NonlinearHeat<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  triangulation.refine_global(5);
  std::cout << "Cells: " << triangulation.n_active_cells() << "\n";
}

template <int dim>
void NonlinearHeat<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "DoFs: " << dof_handler.n_dofs() << "\n";

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler, 0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity.copy_from(dsp);

  M.reinit(sparsity);
  K.reinit(sparsity);
  J.reinit(sparsity);

  u.reinit(dof_handler.n_dofs());
  u_old.reinit(dof_handler.n_dofs());
  du.reinit(dof_handler.n_dofs());
  R.reinit(dof_handler.n_dofs());

  // initial condition: small bump (optional)
  VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), u_old);
  u = u_old;
}

template <int dim>
void NonlinearHeat<dim>::assemble_MK()
{
  M = 0.0; K = 0.0;
  FEValues<dim> fev(fe, quadrature, update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q = quadrature.size();
  FullMatrix<double> cell_M(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_K(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fev.reinit(cell);
    cell_M = 0.0; cell_K = 0.0;

    for (unsigned int q=0;q<n_q;++q)
      for (unsigned int i=0;i<dofs_per_cell;++i)
      {
        const double phi_i = fev.shape_value(i,q);
        const Tensor<1,dim> gi = fev.shape_grad(i,q);
        for (unsigned int j=0;j<dofs_per_cell;++j)
        {
          const double phi_j = fev.shape_value(j,q);
          const Tensor<1,dim> gj = fev.shape_grad(j,q);
          cell_M(i,j) += (phi_i*phi_j)*fev.JxW(q);
          cell_K(i,j) += (gi*gj)*fev.JxW(q);
        }
      }

    cell->get_dof_indices(local);
    constraints.distribute_local_to_global(cell_M, local, M);
    constraints.distribute_local_to_global(cell_K, local, K);
  }
}

// Assemble -Residual into R and Jacobian into J for Newton at current u
template <int dim>
void NonlinearHeat<dim>::assemble_residual_and_jacobian()
{
  J = 0.0;
  R = 0.0;

  // Start with linear part:
  // Residual: (M + dt*kappa*K)u - M u_old + dt * M(u^3)
  // We store RHS as -Residual, so we compute R = -(Residual).
  //
  // Linear contribution: -( (M + dt*kappa*K)u - M u_old ) = (M u_old) - (M + dt*kappa*K)u
  Vector<double> tmp(u.size());
  M.vmult(R, u_old); // R = M u_old

  M.vmult(tmp, u);
  R.add(-1.0, tmp);  // R -= M u

  K.vmult(tmp, u);
  R.add(-dt*kappa, tmp); // R -= dt*kappa*K*u

  // Jacobian linear part: J = (M + dt*kappa*K) + dt * nonlinear_term
  J.copy_from(M);
  J.add(dt*kappa, K);

  // Nonlinear contributions assembled cell-wise: dt * ∫ phi_i * (u^3) dX in residual
  // and dt * ∫ phi_i * (3 u^2) * phi_j dX in Jacobian.
  FEValues<dim> fev(fe, quadrature, update_values | update_JxW_values);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q = quadrature.size();
  Vector<double> cell_rhs(dofs_per_cell);
  FullMatrix<double> cell_Jnl(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local(dofs_per_cell);

  std::vector<double> u_q(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fev.reinit(cell);
    cell_rhs = 0.0;
    cell_Jnl = 0.0;

    fev.get_function_values(u, u_q);

    for (unsigned int q=0;q<n_q;++q)
    {
      const double uq = u_q[q];
      const double u3 = uq*uq*uq;
      const double d  = 3.0*uq*uq; // derivative of u^3

      for (unsigned int i=0;i<dofs_per_cell;++i)
      {
        const double phi_i = fev.shape_value(i,q);

        // Residual contribution stored as RHS: -dt * ∫ phi_i * u^3
        cell_rhs(i) += (-dt) * (phi_i * u3) * fev.JxW(q);

        for (unsigned int j=0;j<dofs_per_cell;++j)
        {
          const double phi_j = fev.shape_value(j,q);
          cell_Jnl(i,j) += (dt) * (phi_i * d * phi_j) * fev.JxW(q);
        }
      }
    }

    cell->get_dof_indices(local);
    constraints.distribute_local_to_global(cell_Jnl, local, J);
    constraints.distribute_local_to_global(cell_rhs, local, R);
  }

  constraints.condense(J);
  constraints.condense(R);
}

template <int dim>
bool NonlinearHeat<dim>::solve_newton()
{
  // Newton loop at a single timestep
  double r0 = -1.0;

  for (unsigned int it=0; it<max_newton; ++it)
  {
    assemble_residual_and_jacobian();
    const double rnorm = R.l2_norm();
    if (it == 0) r0 = rnorm;

    std::cout << "  Newton it " << it << " |R|=" << rnorm << "\n";
    if (rnorm < std::max(tol_abs, tol_rel*r0))
      return true;

    // Solve J du = R (since R is already -Residual)
    SolverControl control(4000, 1e-14 * rnorm);
    SolverCG<Vector<double>> solver(control);

    PreconditionSSOR<SparseMatrix<double>> pre;
    pre.initialize(J, 1.2);

    solver.solve(J, du, R, pre);
    constraints.distribute(du);

    u += du;
    constraints.distribute(u);
  }
  return false;
}

template <int dim>
void NonlinearHeat<dim>::output(const unsigned int step) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(u, "u");
  data_out.build_patches();

  const std::string fn = "u_" + Utilities::int_to_string(step, 3) + ".vtu";
  std::ofstream out(fn);
  data_out.write_vtu(out);
  std::cout << "Wrote " << fn << "\n";
}

template <int dim>
void NonlinearHeat<dim>::run()
{
  make_grid();
  setup_system();
  assemble_MK();

  unsigned int step = 0;
  double time = 0.0;
  output(step);

  while (time < T - 1e-14)
  {
    time += dt;
    ++step;

    // initial guess for Newton: previous solution
    u = u_old;

    const bool ok = solve_newton();
    if (!ok)
    {
      std::cout << "Newton failed at step " << step << "\n";
      break;
    }

    u_old = u;
    if (step % 5 == 0) output(step);
  }
}

int main()
{
  try
  {
    NonlinearHeat<2> p(1);
    p.run();
  }
  catch (std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

