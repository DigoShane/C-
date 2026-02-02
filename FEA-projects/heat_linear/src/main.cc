#include <deal.II/base/function.h>
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
class RHS : public Function<dim>
{
public:
  RHS() : Function<dim>(1) {}
  double value(const Point<dim> &, const unsigned int = 0) const override
  {
    return 0.0; // set a forcing if desired
  }
};

template <int dim>
class HeatLinear
{
public:
  explicit HeatLinear(const unsigned int degree);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_matrices();
  void assemble_rhs(const double time);
  void solve_time_step();
  void output(const unsigned int step) const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity;
  SparseMatrix<double> mass_matrix;     // M
  SparseMatrix<double> stiffness_matrix;// K
  SparseMatrix<double> system_matrix;   // M + dt*kappa*K

  Vector<double> solution;      // u^{n+1}
  Vector<double> old_solution;  // u^{n}
  Vector<double> system_rhs;

  QGauss<dim> quadrature;

  double kappa = 1.0;
  double dt    = 1e-2;
  double T     = 0.1;
};

template <int dim>
HeatLinear<dim>::HeatLinear(const unsigned int degree)
  : fe(degree), dof_handler(triangulation), quadrature(degree + 1)
{}

template <int dim>
void HeatLinear<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  triangulation.refine_global(5);
  std::cout << "Cells: " << triangulation.n_active_cells() << "\n";
}

template <int dim>
void HeatLinear<dim>::setup_system()
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

  mass_matrix.reinit(sparsity);
  stiffness_matrix.reinit(sparsity);
  system_matrix.reinit(sparsity);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // initial condition: u(x,0) = sin(pi x) sin(pi y)
  VectorTools::interpolate(dof_handler,
                           Functions::ZeroFunction<dim>(),
                           old_solution);
  solution = old_solution;
}

template <int dim>
void HeatLinear<dim>::assemble_matrices()
{
  mass_matrix = 0.0;
  stiffness_matrix = 0.0;

  FEValues<dim> fe_values(fe, quadrature,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q = quadrature.size();

  FullMatrix<double> cell_M(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_K(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_M = 0.0;
    cell_K = 0.0;

    for (unsigned int q=0; q<n_q; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i,q);
        const Tensor<1,dim> grad_i = fe_values.shape_grad(i,q);

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          const double phi_j = fe_values.shape_value(j,q);
          const Tensor<1,dim> grad_j = fe_values.shape_grad(j,q);

          cell_M(i,j) += (phi_i * phi_j) * fe_values.JxW(q);
          cell_K(i,j) += (grad_i * grad_j) * fe_values.JxW(q);
        }
      }

    cell->get_dof_indices(local);
    constraints.distribute_local_to_global(cell_M, local, mass_matrix);
    constraints.distribute_local_to_global(cell_K, local, stiffness_matrix);
  }
}

template <int dim>
void HeatLinear<dim>::assemble_rhs(const double time)
{
  system_rhs = 0.0;

  // RHS = M u^n + dt * F^{n+1}
  mass_matrix.vmult(system_rhs, old_solution);

  RHS<dim> rhs_fun;
  rhs_fun.set_time(time);

  Vector<double> f_vec(dof_handler.n_dofs());
  VectorTools::create_right_hand_side(dof_handler, quadrature, rhs_fun, f_vec);

  system_rhs.add(dt, f_vec);

  constraints.condense(system_rhs);

  // system_matrix = M + dt*kappa*K
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(dt * kappa, stiffness_matrix);
  constraints.condense(system_matrix);
}

template <int dim>
void HeatLinear<dim>::solve_time_step()
{
  SolverControl control(4000, 1e-12 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(control);

  PreconditionSSOR<SparseMatrix<double>> pre;
  pre.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, pre);
  constraints.distribute(solution);
}

template <int dim>
void HeatLinear<dim>::output(const unsigned int step) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.build_patches();

  const std::string fn = "u_" + Utilities::int_to_string(step, 3) + ".vtu";
  std::ofstream out(fn);
  data_out.write_vtu(out);
  std::cout << "Wrote " << fn << "\n";
}

template <int dim>
void HeatLinear<dim>::run()
{
  make_grid();
  setup_system();
  assemble_matrices();

  unsigned int step = 0;
  double time = 0.0;

  output(step);

  while (time < T - 1e-14)
  {
    time += dt;
    ++step;

    assemble_rhs(time);
    solve_time_step();

    old_solution = solution;
    if (step % 5 == 0) output(step);
  }
}

int main()
{
  try
  {
    HeatLinear<2> p(1);
    p.run();
  }
  catch (std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

