#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <cmath>


using namespace dealii;

constexpr int dim = 2;

class InitialConcentration : public Function<dim>
{
public:
  InitialConcentration() : Function<dim>(1) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override
  {
    double r = p.norm();
    return 0.5 * std::exp(-10.0 * r * r) + 0.1;
  }
};

class ChemoMechanical
{
public:
  ChemoMechanical();
  void run();

private:
  void setup_mesh();

  void setup_diffusion();
  void assemble_diffusion();
  void solve_diffusion();

  void setup_elasticity();
  void assemble_elasticity();
  void solve_elasticity();

  void output_results(unsigned int step) const;

  Triangulation<dim> triangulation;

  // Diffusion
  FE_Q<dim> fe_c;
  DoFHandler<dim> dof_handler_c;
  AffineConstraints<double> constraints_c;
  SparsityPattern sparsity_c;
  SparseMatrix<double> matrix_c;
  Vector<double> solution_c, old_solution_c, rhs_c;

  // Elasticity
  FESystem<dim> fe_u;
  DoFHandler<dim> dof_handler_u;
  AffineConstraints<double> constraints_u;
  SparsityPattern sparsity_u;
  SparseMatrix<double> matrix_u;
  Vector<double> solution_u, rhs_u;

  // Parameters
  const double D      = 1e-1;
  const double dt     = 1e-2;
  const double E      = 1.0;
  const double nu     = 0.3;
  const double beta   = 0.1;

  double lambda, mu;
};

/* ---------------- Constructor ---------------- */

ChemoMechanical::ChemoMechanical()
  : fe_c(1),
    dof_handler_c(triangulation),
    fe_u(FE_Q<dim>(1), dim),
    dof_handler_u(triangulation)
{
  lambda = E*nu/((1+nu)*(1-2*nu));
  mu     = E/(2*(1+nu));
}

/* ---------------- Mesh ---------------- */

void ChemoMechanical::setup_mesh()
{
  GridGenerator::hyper_ball(triangulation);
  triangulation.refine_global(4);
}

/* ---------------- Diffusion Setup ---------------- */

void ChemoMechanical::setup_diffusion()
{
  dof_handler_c.distribute_dofs(fe_c);

  constraints_c.clear();
  constraints_c.close();

  DynamicSparsityPattern dsp(dof_handler_c.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler_c, dsp);
  sparsity_c.copy_from(dsp);

  matrix_c.reinit(sparsity_c);

  solution_c.reinit(dof_handler_c.n_dofs());
  old_solution_c.reinit(dof_handler_c.n_dofs());
  rhs_c.reinit(dof_handler_c.n_dofs());

  // Gaussian bump at center instead of uniform
  InitialConcentration initial_condition;

  VectorTools::interpolate(dof_handler_c,
                         initial_condition,
                         solution_c);

  old_solution_c = solution_c;
}

/* ---------------- Diffusion Assembly ---------------- */

void ChemoMechanical::assemble_diffusion()
{
  matrix_c = 0;
  rhs_c    = 0;

  QGauss<dim> quad(fe_c.degree + 1);

  FEValues<dim> fe_values(fe_c, quad,
                          update_values |
                          update_gradients |
                          update_JxW_values);

  const unsigned int dofs_per_cell = fe_c.n_dofs_per_cell();
  const unsigned int n_q_points    = quad.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // storage for c^n evaluated at quadrature points
  std::vector<double> c_old_values(n_q_points);

  for (const auto &cell : dof_handler_c.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs    = 0;

    cell->get_dof_indices(local_dof_indices);

    // Evaluate old solution at quadrature points
    fe_values.get_function_values(old_solution_c, c_old_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double JxW = fe_values.JxW(q);
      const double c_old_q = c_old_values[q];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i =
            fe_values.shape_grad(i, q);

        // ----- Mass term RHS: (c^n / dt, phi_i) -----
        cell_rhs(i) += (c_old_q / dt) * phi_i * JxW;

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const double phi_j = fe_values.shape_value(j, q);
          const Tensor<1, dim> grad_phi_j =
              fe_values.shape_grad(j, q);

          // ----- Backward Euler diffusion matrix -----
          cell_matrix(i, j) +=
              ( (phi_i * phi_j) / dt
                + D * grad_phi_i * grad_phi_j )
              * JxW;
        }
      }
    }

    constraints_c.distribute_local_to_global(
        cell_matrix,
        cell_rhs,
        local_dof_indices,
        matrix_c,
        rhs_c);
  }
}


/* ---------------- Diffusion Solve ---------------- */

void ChemoMechanical::solve_diffusion()
{
  SolverControl control(5000,1e-5,true,true);
  SolverCG<> solver(control);

  solver.solve(matrix_c, solution_c, rhs_c,
               PreconditionIdentity());

  std::cout << "||c||_2      = " << solution_c.l2_norm() << std::endl;
  std::cout << "||c||_inf    = " << solution_c.linfty_norm() << std::endl;
  std::cout << "-----------------------------" << std::endl;

  std::cout << "Diffusion CG iterations: "
            << control.last_step()
            << "  residual: "
            << control.last_value()
            << std::endl;
}

/* ---------------- Elasticity Setup ---------------- */

void ChemoMechanical::setup_elasticity()
{
  dof_handler_u.distribute_dofs(fe_u);

  constraints_u.clear();

  //Clamp boundary.
  VectorTools::interpolate_boundary_values(
      dof_handler_u,
      0,   // boundary id (hyper_ball uses 0)
      Functions::ZeroFunction<dim>(dim),
      constraints_u);
  
  constraints_u.close();

  DynamicSparsityPattern dsp(dof_handler_u.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler_u, dsp, constraints_u);
  sparsity_u.copy_from(dsp);

  matrix_u.reinit(sparsity_u);

  solution_u.reinit(dof_handler_u.n_dofs());
  rhs_u.reinit(dof_handler_u.n_dofs());
}

/* ---------------- Elasticity Assembly ---------------- */

void ChemoMechanical::assemble_elasticity()
{
  matrix_u = 0;
  rhs_u    = 0;

  QGauss<dim> quad(fe_u.degree+1);
  FEValues<dim> fe_values_u(fe_u, quad,
      update_values | update_gradients | update_JxW_values);

  FEValues<dim> fe_values_c(fe_c, quad,
      update_values);

  const unsigned int dofs_per_cell = fe_u.n_dofs_per_cell();
  const unsigned int n_q = quad.size();

  FullMatrix<double> cell_matrix(dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

  std::vector<double> c_values(n_q);

  for (auto cell_u : dof_handler_u.active_cell_iterators())
  {
    fe_values_u.reinit(cell_u);
    cell_matrix = 0;
    cell_rhs    = 0;

    cell_u->get_dof_indices(local_dofs);

    auto cell_c = dof_handler_c.begin_active();
    std::advance(cell_c, cell_u->active_cell_index());
    fe_values_c.reinit(cell_c);
    fe_values_c.get_function_values(solution_c, c_values);

    for (unsigned int q=0;q<n_q;++q)
    {
      const double c = c_values[q];

      SymmetricTensor<2,dim> eigenstrain =
          beta*c*unit_symmetric_tensor<dim>();

      for (unsigned int i=0;i<dofs_per_cell;++i)
      {
        // Get gradient of shape function i (vector-valued)
        const unsigned int component_i =
    fe_u.system_to_component_index(i).first;

    Tensor<2,dim> grad_i;
    Tensor<1,dim> grad_phi_i =
    fe_values_u.shape_grad(i,q);

    for (unsigned int d=0; d<dim; ++d)
        grad_i[component_i][d] = grad_phi_i[d];

    SymmetricTensor<2,dim> eps_i =
        symmetrize(grad_i);

        for (unsigned int j=0;j<dofs_per_cell;++j)
        {
          // Get gradient of shape function j (vector-valued)
          const unsigned int component_j =
          fe_u.system_to_component_index(j).first;
      
          Tensor<2,dim> grad_j;
          Tensor<1,dim> grad_phi_j =
              fe_values_u.shape_grad(j,q);
          
          for (unsigned int d=0; d<dim; ++d)
              grad_j[component_j][d] = grad_phi_j[d];
          
          SymmetricTensor<2,dim> eps_j =
              symmetrize(grad_j);

          cell_matrix(i,j) +=
            (lambda*trace(eps_i)*trace(eps_j)
             + 2*mu*(eps_i*eps_j))
            * fe_values_u.JxW(q);
        }

        cell_rhs(i) +=
          (lambda*trace(eigenstrain)*trace(eps_i)
           + 2*mu*(eigenstrain*eps_i))
          * fe_values_u.JxW(q);
      }
    }

    constraints_u.distribute_local_to_global(
        cell_matrix, cell_rhs,
        local_dofs,
        matrix_u, rhs_u);
  }
}

/* ---------------- Elasticity Solve ---------------- */

void ChemoMechanical::solve_elasticity()
{
  SolverControl control(5000,1e-5,true,true);
  SolverCG<> solver(control);

  solver.solve(matrix_u, solution_u, rhs_u,
               PreconditionIdentity());

  std::cout << "Elasticity CG iterations: "
            << control.last_step()
            << "  residual: "
            << control.last_value()
            << std::endl;

  double max_mag = 0.0;

  for (unsigned int i = 0; i < solution_u.size(); i += dim)
  {
    double ux = solution_u[i];
    double uy = solution_u[i+1];
    double mag = std::sqrt(ux*ux + uy*uy);
    max_mag = std::max(max_mag, mag);
  }
  
  std::cout << "max |u| magnitude = "
            << max_mag << std::endl;


  std::cout << "||u||_2      = " << solution_u.l2_norm() << std::endl;
  std::cout << "||u||_inf    = " << solution_u.linfty_norm() << std::endl;
  std::cout << "max |u| comp = " << solution_u.linfty_norm() << std::endl;
  std::cout << "-----------------------------" << std::endl;

}

/* ---------------- Output ---------------- */

void ChemoMechanical::output_results(unsigned int step) const
{
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_c);
    data_out.add_data_vector(solution_c, "concentration");
    data_out.build_patches();

    std::ofstream out("concentration-" +
                      std::to_string(step) + ".vtu");
    data_out.write_vtu(out);
  }

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_u);
    data_out.add_data_vector(solution_u, "displacement");
    data_out.build_patches();

    std::ofstream out("displacement-" +
                      std::to_string(step) + ".vtu");
    data_out.write_vtu(out);
  }
}

/* ---------------- Run ---------------- */

void ChemoMechanical::run()
{
  setup_mesh();
  setup_diffusion();
  setup_elasticity();

  const unsigned int n_steps = 200;

  for (unsigned int t=0;t<n_steps;++t)
  {
    std::cout << "Time step " << t << std::endl;

    assemble_diffusion();
    solve_diffusion();
    old_solution_c = solution_c;

    assemble_elasticity();
    solve_elasticity();

    output_results(t);
  }
}

/* ---------------- Main ---------------- */

int main()
{
  try
  {
    ChemoMechanical problem;
    problem.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << exc.what() << std::endl;
    return 1;
  }
  return 0;
}

