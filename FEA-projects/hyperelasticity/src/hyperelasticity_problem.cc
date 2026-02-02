#include "hyperelasticity_problem.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iostream>

namespace dealii_hyper
{
  using namespace dealii;

  // ===========================================================================
  // Nonlinear FEA example: 2D compressible Neo-Hookean hyperelasticity
  //
  // Kinematics:
  //   F = I + Grad(u)  (deformation gradient)
  //   J = det(F)
  //
  // Material model (compressible Neo-Hookean, total Lagrangian form):
  //   First Piola stress:
  //     P = μ (F - F^{-T}) + λ ln(J) F^{-T}
  //
  // Weak form (residual):
  //   R(u; v) = ∫_Ω Grad(v) : P(F(u)) dX  -  ∫_{Γ_N} v · t dS
  //
  // Newton linearization:
  //   J(u_k) δu = -R(u_k)
  //   u_{k+1} = u_k + δu
  //
  // Load stepping:
  //   Apply traction scaled by alpha in [0,1] over several steps for robustness.
  // ===========================================================================


  template <int dim>
  HyperElasticityProblem<dim>::HyperElasticityProblem(const unsigned int degree)
    // Vector-valued FE for displacement: dim components (ux,uy in 2D)
    : fe(FE_Q<dim>(degree), dim)
    // DoFHandler must be constructed with a Triangulation reference
    , dof_handler(triangulation)
    // Quadrature order: degree+1 is a safe baseline
    , quadrature(degree + 1)
    , face_quadrature(degree + 1)
  {
    // Choose E and nu and convert to Lamé constants for compressible behavior.
    const double E  = 1.0;
    const double nu = 0.3;

    mu     = E / (2.0 * (1.0 + nu));
    lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));

    // Reference traction magnitude. If Newton struggles, reduce this.
    traction_T = 1e-2;
  }


  template <int dim>
  void HyperElasticityProblem<dim>::make_grid()
  {
    // Structured mesh: Nx by Ny subdivisions
    const unsigned int Nx = 40;
    const unsigned int Ny = 8;

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              {Nx, Ny},
                                              Point<dim>(0.0, 0.0),
                                              Point<dim>(L, H),
                                              /*colorize*/ true);

    // Explicitly tag left and right boundaries by geometry
    set_boundary_ids();

    std::cout << "Cells: " << triangulation.n_active_cells() << "\n";
  }


  template <int dim>
  void HyperElasticityProblem<dim>::set_boundary_ids()
  {
    // Boundary ID assignment:
    //   x=0 -> id=1 (Dirichlet clamp)
    //   x=L -> id=2 (Neumann traction)
    const double tol = 1e-12;

    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
        {
          const auto c = face->center();
          if (std::abs(c[0] - 0.0) < tol)
            face->set_boundary_id(1);
          else if (std::abs(c[0] - L) < tol)
            face->set_boundary_id(2);
        }
  }


  template <int dim>
  void HyperElasticityProblem<dim>::setup_system()
  {
    // Assign global DoF indices for the chosen FE space on this mesh.
    dof_handler.distribute_dofs(fe);
    std::cout << "DoFs: " << dof_handler.n_dofs() << "\n";

    // Constraints store Dirichlet BCs (and can store other linear constraints).
    constraints.clear();

    // Clamp left boundary (boundary id=1): u=0 (all components)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints);
    constraints.close();

    // Build sparsity pattern for the tangent stiffness matrix.
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    // Allocate sparse matrix and vectors.
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  // Compute P and dP/dF for compressible Neo-Hookean.
  //
  // P = μ (F - F^{-T}) + λ ln(J) F^{-T}
  //
  // Tangent used here (common compressible Neo-Hookean form in TL):
  //   dP_{iJ}/dF_{kL} =
  //     μ δ_{ik} δ_{JL}
  //     + (μ - λ ln J) A_{kJ} A_{iL}
  //     + λ A_{kL} A_{iJ}
  // where A = F^{-T}.
  template <int dim>
  void HyperElasticityProblem<dim>::compute_P_and_tangent(const Tensor<2, dim> &F,
                                                         Tensor<2, dim>       &P,
                                                         Tensor<4, dim>       &dPdF) const
  {
    const double J = determinant(F);
    if (!(J > 0.0))
      throw std::runtime_error("Nonpositive det(F). Load too large or mesh too coarse.");

    const Tensor<2, dim> Finv = invert(F);
    const Tensor<2, dim> A    = transpose(Finv); // F^{-T}
    const double lnJ          = std::log(J);

    // First Piola stress
    P = mu * (F - A) + lambda * lnJ * A;

    // Tangent tensor
    dPdF.clear();
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int Jidx = 0; Jidx < dim; ++Jidx)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int Lidx = 0; Lidx < dim; ++Lidx)
          {
            const double delta_ik = (i == k) ? 1.0 : 0.0;
            const double delta_JL = (Jidx == Lidx) ? 1.0 : 0.0;

            dPdF[i][Jidx][k][Lidx] =
              mu * delta_ik * delta_JL
              + (mu - lambda * lnJ) * A[k][Jidx] * A[i][Lidx]
              + lambda * A[k][Lidx] * A[i][Jidx];
          }
  }


  template <int dim>
  void HyperElasticityProblem<dim>::assemble_system(const double load_factor)
  {
    // Reset tangent matrix and residual (stored as RHS = -R)
    system_matrix = 0.0;
    system_rhs    = 0.0;

    FEValues<dim> fe_values(fe,
                            quadrature,
                            update_values | update_gradients | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature,
                                     update_values | update_JxW_values |
                                       update_quadrature_points);

    // Extractor: treat FE components [0..dim-1] as one vector field
    const FEValuesExtractors::Vector u(0);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q           = quadrature.size();
    const unsigned int n_qf          = face_quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Neumann traction on right boundary, scaled by load_factor
    RightTraction<dim> traction(load_factor * traction_T);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      // Compute Grad(u) at each quadrature point from the current solution iterate
      // deal.II API: get_function_gradients(solution_vector, output_array)
      std::vector<Tensor<2, dim>> grad_u(n_q);
      fe_values[u].get_function_gradients(solution, grad_u);

      // ---------------- Volume contributions: residual and Jacobian ----------------
      for (unsigned int q = 0; q < n_q; ++q)
      {
        // F = I + Grad(u)
        Tensor<2, dim> F;
        for (unsigned int i = 0; i < dim; ++i)
          F[i][i] = 1.0;
        F += grad_u[q];

        Tensor<2, dim> P;
        Tensor<4, dim> dPdF;
        compute_P_and_tangent(F, P, dPdF);

        // Assemble cell residual and tangent using:
        //   R_i = ∫ Grad(N_i) : P dX  - traction_term
        //   K_ij = ∫ Grad(N_i) : (dP/dF : Grad(N_j)) dX
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<2, dim> GradNi = fe_values[u].gradient(i, q);

          // Residual contribution (scalar)
          double Ri = 0.0;
          for (unsigned int a = 0; a < dim; ++a)
            for (unsigned int A = 0; A < dim; ++A)
              Ri += GradNi[a][A] * P[a][A];

          // We store RHS as -R, so subtract Ri
          cell_rhs(i) -= Ri * fe_values.JxW(q);

          // Tangent contribution
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const Tensor<2, dim> GradNj = fe_values[u].gradient(j, q);

            double Kij = 0.0;
            for (unsigned int a = 0; a < dim; ++a)
              for (unsigned int A = 0; A < dim; ++A)
                for (unsigned int k = 0; k < dim; ++k)
                  for (unsigned int Lidx = 0; Lidx < dim; ++Lidx)
                    Kij += GradNi[a][A] * dPdF[a][A][k][Lidx] * GradNj[k][Lidx];

            cell_matrix(i, j) += Kij * fe_values.JxW(q);
          }
        }
      }

      // ---------------- Neumann traction contribution on boundary id=2 ----------------
      // In weak form, traction appears as: -∫ v · t dS
      // Since system_rhs stores -R, the traction contributes +∫ v · t dS to RHS.
      for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        if (cell->face(face_no)->at_boundary() &&
            cell->face(face_no)->boundary_id() == 2)
        {
          fe_face_values.reinit(cell, face_no);

          for (unsigned int q = 0; q < n_qf; ++q)
          {
            const Tensor<1, dim> t = traction.value(fe_face_values.quadrature_point(q));
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> Ni = fe_face_values[u].value(i, q);
              cell_rhs(i) += (Ni * t) * fe_face_values.JxW(q);
            }
          }
        }

      // Copy local contributions into global system while applying constraints.
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
    }
  }


  template <int dim>
  bool HyperElasticityProblem<dim>::solve_newton_step()
  {
    // Solve linearized Newton system:
    //   K * delta_u = system_rhs  (where system_rhs = -R)
    SolverControl control(4000, 1e-14 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);

    // Enforce constraints on the update (important for Dirichlet BCs).
    constraints.distribute(newton_update);
    return true;
  }


  template <int dim>
  bool HyperElasticityProblem<dim>::newton_solve(const double load_factor)
  {
    // Newton loop at fixed load_factor.
    double rhs0 = -1.0;

    for (unsigned int it = 0; it < max_newton_iters; ++it)
    {
      // Assemble Jacobian K and RHS (-R) at current solution
      assemble_system(load_factor);

      const double rhs_norm = system_rhs.l2_norm();
      if (it == 0)
        rhs0 = rhs_norm;

      std::cout << "  Newton it " << it << " |R|=" << rhs_norm << "\n";

      // Convergence check (relative to initial residual and absolute floor)
      if (rhs_norm < std::max(newton_tol_abs, newton_tol_rel * rhs0))
      {
        std::cout << "  Converged.\n";
        return true;
      }

      // Solve for delta_u and update solution
      solve_newton_step();
      solution += newton_update;

      // Enforce Dirichlet values exactly after update.
      constraints.distribute(solution);
    }

    std::cout << "  Newton failed to converge.\n";
    return false;
  }


  template <int dim>
  void HyperElasticityProblem<dim>::output_results(const unsigned int step) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    // Important: pass explicit vector-of-strings + interpretation
    // to avoid ambiguous overloads and mark the data as a vector field.
    const std::vector<std::string> names = {"u_x", "u_y"};
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(solution,
                             names,
                             DataOut<dim>::type_dof_data,
                             interpretation);

    data_out.build_patches();

    const std::string filename =
      "solution_step_" + Utilities::int_to_string(step, 2) + ".vtu";

    std::ofstream out(filename);
    data_out.write_vtu(out);

    std::cout << "Wrote: " << filename << "\n";
  }


  template <int dim>
  void HyperElasticityProblem<dim>::run()
  {
    make_grid();
    setup_system();

    // Continuation: ramp traction from 0 → 1 over n_steps.
    // This is standard practice for nonlinear mechanics.
    const unsigned int n_steps = 5;

    for (unsigned int step = 0; step <= n_steps; ++step)
    {
      const double load_factor = static_cast<double>(step) / static_cast<double>(n_steps);

      std::cout << "\nLoad step " << step << "/" << n_steps
                << " (factor=" << load_factor << ")\n";

      const bool ok = newton_solve(load_factor);
      output_results(step);

      if (!ok)
      {
        std::cout << "Stopping due to Newton failure.\n";
        break;
      }
    }
  }

  // Explicit instantiation for dim=2 (keeps linkers happy in multi-file builds).
  template class HyperElasticityProblem<2>;

} // namespace dealii_hyper

