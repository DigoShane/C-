#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class InitialCondition : public Function<2>
{
public:
  double value(const Point<2> &p,
               const unsigned int = 0) const override
  {
    return 1.0 + 0.5 * std::exp(-5.0 * p.square());
  }
};

class LithiumDiffusion
{
public:
  LithiumDiffusion();
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve_time_step();
  void output_results(const unsigned int time_step) const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> old_solution;
  Vector<double> system_rhs;

  double time;
  double time_step;
  const double D = 1.0;
};

LithiumDiffusion::LithiumDiffusion()
  : fe(1),
    dof_handler(triangulation),
    time(0.0),
    time_step(1.0)
{}

void LithiumDiffusion::setup_system()
{
  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // Set initial condition
  VectorTools::interpolate(dof_handler,
                           InitialCondition(),
                           old_solution);
}

void LithiumDiffusion::assemble_system()
{
  system_matrix = 0;
  system_rhs    = 0;

  QGauss<2> quadrature_formula(2);
  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values |
                        update_gradients |
                        update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs    = 0;

    cell->get_dof_indices(local_dof_indices);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // Mass term
          cell_matrix(i, j) +=
              fe_values.shape_value(i, q) *
              fe_values.shape_value(j, q) *
              fe_values.JxW(q);

          // Diffusion term (implicit Euler)
          cell_matrix(i, j) +=
              time_step * D *
              fe_values.shape_grad(i, q) *
              fe_values.shape_grad(j, q) *
              fe_values.JxW(q);
        }

        // RHS = M * old_solution
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_rhs(i) +=
              fe_values.shape_value(i, q) *
              old_solution(local_dof_indices[j]) *
              fe_values.shape_value(j, q) *
              fe_values.JxW(q);
        }
      }
    }

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        system_matrix.add(local_dof_indices[i],
                          local_dof_indices[j],
                          cell_matrix(i, j));
      }

      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }
}

void LithiumDiffusion::solve_time_step()
{
  SolverControl solver_control(1000, 1e-12);
  SolverCG<> solver(solver_control);

  solver.solve(system_matrix,
               solution,
               system_rhs,
               PreconditionIdentity());
}

void LithiumDiffusion::output_results(const unsigned int timestep) const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "concentration");
  data_out.build_patches();

  std::ofstream output("solution-" +
                       std::to_string(timestep) +
                       ".vtu");

  data_out.write_vtu(output);
}

void LithiumDiffusion::run()
{
  GridGenerator::hyper_ball(triangulation);
  triangulation.refine_global(4);

  setup_system();

  const unsigned int n_time_steps = 50;

  for (unsigned int t = 0; t < n_time_steps; ++t)
  {
    assemble_system();
    solve_time_step();

    old_solution = solution;
    time += time_step;

    output_results(t);
  }
}

int main()
{
  LithiumDiffusion problem;
  problem.run();
}

