#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class InitialCondition : public Function<2>
{
public:
  double value(const Point<2> &p,
               const unsigned int = 0) const override
  {
    return 0.05 * std::sin(6*p[0]) * std::sin(6*p[1]);
  }
};

class CahnHilliard
{
public:
  CahnHilliard();
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve_time_step();
  void output_results(unsigned int t) const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> old_solution;
  Vector<double> system_rhs;

  const double dt = 1e-3;
  const double kappa = 1e-2;
};

CahnHilliard::CahnHilliard()
  : fe(2),
    dof_handler(triangulation)
{}

void CahnHilliard::setup_system()
{
  GridGenerator::hyper_ball(triangulation);
  triangulation.refine_global(4);

  dof_handler.distribute_dofs(fe);

  constraints.clear();
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,dsp,constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  VectorTools::interpolate(dof_handler,
                           InitialCondition(),
                           solution);

  old_solution = solution;
}

void CahnHilliard::assemble_system()
{
  system_matrix = 0;
  system_rhs    = 0;

  QGauss<2> quadrature(fe.degree + 1);
  FEValues<2> fe_values(fe, quadrature,
                        update_values |
                        update_gradients |
                        update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q = quadrature.size();

  FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

  std::vector<double> c_old_values(n_q);
  std::vector<Tensor<1,2>> grad_c_old(n_q);

  for (auto cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;
    cell->get_dof_indices(local_dofs);

    fe_values.get_function_values(old_solution,c_old_values);
    fe_values.get_function_gradients(old_solution,grad_c_old);

    for (unsigned int q=0;q<n_q;++q)
    {
      double c_old = c_old_values[q];
      Tensor<1,2> grad_c = grad_c_old[q];

      for (unsigned int i=0;i<dofs_per_cell;++i)
      {
        double phi_i = fe_values.shape_value(i,q);
        Tensor<1,2> grad_phi_i = fe_values.shape_grad(i,q);

        for (unsigned int j=0;j<dofs_per_cell;++j)
        {
          double phi_j = fe_values.shape_value(j,q);
          Tensor<1,2> grad_phi_j =
              fe_values.shape_grad(j,q);

          cell_matrix(i,j) +=
              (phi_i*phi_j/dt
               + grad_phi_i*grad_phi_j
               + kappa*(grad_phi_i*grad_phi_j))
              * fe_values.JxW(q);
        }

        cell_rhs(i) +=
            (c_old/dt * phi_i
             - 3*c_old*c_old * grad_c * grad_phi_i)
            * fe_values.JxW(q);
      }
    }

    constraints.distribute_local_to_global(
        cell_matrix,cell_rhs,
        local_dofs,
        system_matrix,system_rhs);
  }
}

void CahnHilliard::solve_time_step()
{
  SolverControl solver_control(5000,1e-5,true,true);
  SolverCG<> solver(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix,1.2);

  solver.solve(system_matrix,
               solution,
               system_rhs,
               preconditioner);

  std::cout << "CG iterations: "
            << solver_control.last_step()
            << "  final residual: "
            << solver_control.last_value()
            << std::endl;
}

void CahnHilliard::output_results(unsigned int t) const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,"c");
  data_out.build_patches();

  std::ofstream out("solution-"+std::to_string(t)+".vtu");
  data_out.write_vtu(out);
}

void CahnHilliard::run()
{
  setup_system();

  const unsigned int n_steps = 50;

  for (unsigned int t=0;t<n_steps;++t)
  {
    assemble_system();
    solve_time_step();
    old_solution = solution;
    output_results(t);
    std::cout<<"Time step "<<t<<" complete\n";
  }
}

int main()
{
  CahnHilliard problem;
  problem.run();
}

