#include "gl_problem.h"

#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

namespace gl
{
  using namespace dealii;

  template <int dim>
  GLProblem<dim>::GLProblem(const unsigned int degree)
    : fe(FE_Q<dim>(degree), 2) // (u,v) each Q_degree
    , dof_handler(triangulation)
    , quadrature(degree + 1)
    , face_quadrature(degree + 1)
  {}

  template <int dim>
  Tensor<1, dim> GLProblem<dim>::A_field(const Point<dim> &x) const
  {
    static_assert(dim == 2, "This A_field implementation is for dim=2.");

    // Symmetric gauge for uniform B = H:
    // A = (-H y/2, H x/2)
    Tensor<1, dim> A;
    A[0] = -0.5 * H_ * x[1];
    A[1] =  0.5 * H_ * x[0];
    return A;
  }

  template <int dim>
  double GLProblem<dim>::A_sq(const Point<dim> &x) const
  {
    const Tensor<1, dim> A = A_field(x);
    return A * A;
  }

  template <int dim>
  void GLProblem<dim>::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(5); // adjust as needed

    std::cout << "Cells: " << triangulation.n_active_cells() << "\n";
  }

  template <int dim>
  void GLProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    std::cout << "DoFs: " << dof_handler.n_dofs() << "\n";

    constraints.clear();

    // Dirichlet: psi = 1 on entire boundary
    PsiBoundary<dim> boundary_fun;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             /*boundary_id=*/0,
                                             boundary_fun,
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Initial guess: psi = 1 everywhere (consistent with BC)
    VectorTools::interpolate(dof_handler, boundary_fun, solution);
    constraints.distribute(solution);
  }

  template <int dim>
  void GLProblem<dim>::assemble_system()
  {
    system_matrix = 0.0; // J 
    system_rhs    = 0.0; // -R 
			 // J\delta x=-R. x_n+1 = x_n + \delta x

    FEValues<dim> fe_values(fe,
                            quadrature,
                            update_values |
                            update_gradients |
                            update_quadrature_points |
                            update_JxW_values);

    const FEValuesExtractors::Scalar u_fe(0);
    const FEValuesExtractors::Scalar v_fe(1);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q           = quadrature.size();

    //matrices and vectors at the local level to be assembled into the global system.
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Values/gradients of current iterate at quadrature points
    std::vector<double> u_val(n_q), v_val(n_q);
    std::vector<Tensor<1, dim>> grad_u(n_q), grad_v(n_q);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      fe_values[u_fe].get_function_values(solution, u_val);
      fe_values[v_fe].get_function_values(solution, v_val);
      fe_values[u_fe].get_function_gradients(solution, grad_u);
      fe_values[v_fe].get_function_gradients(solution, grad_v);

      for (unsigned int q = 0; q < n_q; ++q)
      {
        const Point<dim> x_q = fe_values.quadrature_point(q);

        const Tensor<1, dim> A = A_field(x_q);
        const double Asq = A_sq(x_q);

        const double u = u_val[q];
        const double v = v_val[q];
        const double s = u*u + v*v;         // |psi|^2
        const double m = Asq + kappa_*kappa_*(s - 1.0);

        // For Newton Jacobian, derivatives of m are:
        // m = |A|^2 + kappa^2 (u^2+v^2-1)
        // dm/du = 2 kappa^2 u
        // dm/dv = 2 kappa^2 v
        const double dm_du = 2.0 * kappa_*kappa_ * u;
        const double dm_dv = 2.0 * kappa_*kappa_ * v;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // Test functions for u and v components
          const double        phi_u_i = fe_values[u_fe].value(i, q);
          const double        phi_v_i = fe_values[v_fe].value(i, q);
          const Tensor<1,dim> grad_phi_u_i = fe_values[u_fe].gradient(i, q);
          const Tensor<1,dim> grad_phi_v_i = fe_values[v_fe].gradient(i, q);

          // Residual contributions (weak form):
          //
          // Ru: ∫ grad(phi_u)·grad(u)  dx  + ∫ [-2 phi_u (A·grad(v)) + m phi_u u] dx
          // Rv: ∫ grad(phi_v)·grad(v)  dx  + ∫ [ 2 phi_v (A·grad(u)) + m phi_v v] dx
          //
          // We store system_rhs = -R, hence the minus signs below.
          double Ru_i = 0.0;
          double Rv_i = 0.0;

          Ru_i += grad_phi_u_i * grad_u[q];
          Ru_i += -2.0 * phi_u_i * (A * grad_v[q]);
          Ru_i += (m * phi_u_i * u);

          Rv_i += grad_phi_v_i * grad_v[q];
          Rv_i +=  2.0 * phi_v_i * (A * grad_u[q]);
          Rv_i += (m * phi_v_i * v);

          cell_rhs(i) -= (Ru_i + Rv_i) * fe_values.JxW(q);

          // Jacobian assembly:
          //
          // For Ru:
          //   dRu/du: ∫ grad(phi_u)·grad(δu) + ∫ phi_u [ m + u*dm/du ] δu
          //   dRu/dv: ∫ -2 phi_u A·grad(δv) + ∫ phi_u [ u*dm/dv ] δv
          //
          // For Rv:
          //   dRv/dv: ∫ grad(phi_v)·grad(δv) + ∫ phi_v [ m + v*dm/dv ] δv
          //   dRv/du: ∫  2 phi_v A·grad(δu) + ∫ phi_v [ v*dm/du ] δu
          //
          // Note: u*dm/du = 2 kappa^2 u^2, v*dm/dv = 2 kappa^2 v^2,
          // and cross term u*dm/dv = v*dm/du = 2 kappa^2 u v.

          const double m_uu = m + u * dm_du; // m + 2 kappa^2 u^2
          const double m_vv = m + v * dm_dv; // m + 2 kappa^2 v^2
          const double m_uv = u * dm_dv;     // 2 kappa^2 u v
          const double m_vu = v * dm_du;     // 2 kappa^2 u v

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // Trial functions
            const double        phi_u_j = fe_values[u_fe].value(j, q);
            const double        phi_v_j = fe_values[v_fe].value(j, q);
            const Tensor<1,dim> grad_phi_u_j = fe_values[u_fe].gradient(j, q);
            const Tensor<1,dim> grad_phi_v_j = fe_values[v_fe].gradient(j, q);

            double Kij = 0.0;

            // Contributions to equation for u-test (phi_u_i)
            // dRu/du
            Kij += (grad_phi_u_i * grad_phi_u_j);
            Kij += (phi_u_i * m_uu * phi_u_j);

            // dRu/dv
            Kij += (-2.0 * phi_u_i * (A * grad_phi_v_j));
            Kij += (phi_u_i * m_uv * phi_v_j);

            // Contributions to equation for v-test (phi_v_i)
            // dRv/dv
            Kij += (grad_phi_v_i * grad_phi_v_j);
            Kij += (phi_v_i * m_vv * phi_v_j);

            // dRv/du
            Kij += ( 2.0 * phi_v_i * (A * grad_phi_u_j));
            Kij += (phi_v_i * m_vu * phi_u_j);

            cell_matrix(i, j) += Kij * fe_values.JxW(q);
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
    }
  }

  template <int dim>
  bool GLProblem<dim>::solve_newton_step()
  {
    // Solve: system_matrix * newton_update = system_rhs
    SolverControl control(4000, 1e-14 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
    constraints.distribute(newton_update);
    return true;
  }

  template <int dim>
  bool GLProblem<dim>::newton_solve()
  {
    double rhs0 = -1.0;

    for (unsigned int it = 0; it < max_newton_iters_; ++it)
    {
      assemble_system();
      const double rnorm = system_rhs.l2_norm(); // this is ||-R|| = ||R||

      if (it == 0)
        rhs0 = rnorm;

      std::cout << "Newton it " << it << " |R| = " << rnorm << "\n";

      if (rnorm < std::max(newton_tol_abs_, newton_tol_rel_ * rhs0))
      {
        std::cout << "Converged.\n";
        return true;
      }

      solve_newton_step();

      solution += newton_update;
      constraints.distribute(solution);

      if (output_every_ > 0 && (it % output_every_ == 0))
        output_results(it);
    }

    std::cout << "Newton failed.\n";
    return false;
  }

  template <int dim>
  void GLProblem<dim>::output_results(const unsigned int it) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    // Name components explicitly
    const std::vector<std::string> names = {"u", "v"};
    data_out.add_data_vector(solution, names);

    data_out.build_patches();

    const std::string filename = "gl_solution_" + Utilities::int_to_string(it, 2) + ".vtu";
    std::ofstream out(filename);
    data_out.write_vtu(out);

    std::cout << "Wrote: " << filename << "\n";
  }

  template <int dim>
  void GLProblem<dim>::run()
  {
    make_grid();
    setup_system();

    const bool ok = newton_solve();
    (void)ok;

    // Final output
    output_results(99);
  }

  template class GLProblem<2>;
} // namespace gl

