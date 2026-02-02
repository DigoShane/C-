#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/full_matrix.h>
#include <cmath>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// ---------------------- Potential V(x) ----------------------
// Start with V = 0. You can replace this with a harmonic trap, barrier, etc.
template <int dim>
class Potential : public Function<dim>
{
public:
  double value(const Point<dim> &, const unsigned int = 0) const override
  {
    return 0.0;
  }
};

// ---------------------- Initial condition psi0 ----------------------
// psi = a + i b. Here: Gaussian in a, b=0.
template <int dim>
class Psi0 : public Function<dim>
{
public:
  Psi0() : Function<dim>(2) {} // 2 components: a,b

  double value(const Point<dim> &p, const unsigned int c) const override
  {
    const double x = p[0] - 0.5;
    const double y = p[1] - 0.5;
    const double g = std::exp(-50.0 * (x * x + y * y));
    return (c == 0) ? g : 0.0;
  }
};

// ---------------------- Time-dependent Schrödinger solver ----------------------
//
// PDE: i ψ_t = -1/2 Δψ + V(x) ψ  in Ω,  ψ=0 on ∂Ω
//
// Split ψ = a + i b:
//   a_t = -H b
//   b_t =  H a
// where H = (1/2)(-Δ) + V.
//
// Crank–Nicolson (CN):
//   a^{n+1} = a^n + dt/2 * ( -H b^{n+1} - H b^n )
//   b^{n+1} = b^n + dt/2 * (  H a^{n+1} + H a^n )
//
// Rearranged linear system:
// [ M        + dt/2 * H ] [ a^{n+1} ] = [ M        - dt/2 * H ] [ a^n ]
// [ -dt/2 * H      M   ] [ b^{n+1} ]   [ +dt/2 * H      M   ] [ b^n ]
//
// (Signs depend on convention; this system is consistent with the split above.)
//
// We assemble M (mass) and H (Hamiltonian operator) in FE form and then build
// these 2×2 block matrices.
//
template <int dim>
class SchrodingerTD
{
public:
  explicit SchrodingerTD(unsigned int degree);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_M_and_H();      // builds M and H blocks (same operator for a,b)
  void build_CN_matrices();     // builds A and B block matrices for CN
  void assemble_rhs();          // rhs = B * psi_old
  void solve_step();            // solve A * psi = rhs
  void output(unsigned int step) const;

  Triangulation<dim> triangulation;

  FESystem<dim>   fe;          // (a,b) as 2 scalar components
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  // Block sizes: [n_a, n_b]
  std::vector<types::global_dof_index> dofs_per_block;

  BlockSparsityPattern      bsp;
  BlockSparseMatrix<double> M;   // mass block matrix (diagonal: M_aa, M_bb)
  BlockSparseMatrix<double> H;   // Hamiltonian operator block matrix (diagonal: H_aa, H_bb)
  BlockSparseMatrix<double> A;   // CN left matrix
  BlockSparseMatrix<double> B;   // CN right matrix

  BlockVector<double> psi;      // [a;b] at new time
  BlockVector<double> psi_old;  // [a;b] at old time
  BlockVector<double> rhs;

  QGauss<dim> quadrature;

  double dt = 1e-3;
  double T  = 1e-2;

  unsigned int output_every = 1;
};

template <int dim>
SchrodingerTD<dim>::SchrodingerTD(unsigned int degree)
  : fe(FE_Q<dim>(degree), 2)
  , dof_handler(triangulation)
  , quadrature(degree + 1)
{}

template <int dim>
void SchrodingerTD<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  triangulation.refine_global(5);
  std::cout << "Cells: " << triangulation.n_active_cells() << "\n";
}

template <int dim>
void SchrodingerTD<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  // Ensure block structure: all a-DoFs first, then all b-DoFs
  DoFRenumbering::component_wise(dof_handler);

  // Count DoFs per component (a and b)
  std::vector<types::global_dof_index> dofs_per_component(2);
  DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);

  dofs_per_block = {dofs_per_component[0], dofs_per_component[1]};
  std::cout << "DoFs: total=" << dof_handler.n_dofs()
            << " (a=" << dofs_per_block[0]
            << ", b=" << dofs_per_block[1] << ")\n";

  constraints.clear();
  // Dirichlet: psi=0 on boundary for both components
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(2),
                                           constraints);
  constraints.close();

  // Build block sparsity pattern
  BlockDynamicSparsityPattern bdsp(2, 2);
  bdsp.block(0,0).reinit(dofs_per_block[0], dofs_per_block[0]);
  bdsp.block(0,1).reinit(dofs_per_block[0], dofs_per_block[1]);
  bdsp.block(1,0).reinit(dofs_per_block[1], dofs_per_block[0]);
  bdsp.block(1,1).reinit(dofs_per_block[1], dofs_per_block[1]);
  bdsp.collect_sizes();

  DoFTools::make_sparsity_pattern(dof_handler, bdsp, constraints, /*keep_constrained_dofs=*/false);

  bsp.copy_from(bdsp);

  M.reinit(bsp);
  H.reinit(bsp);
  A.reinit(bsp);
  B.reinit(bsp);

  psi.reinit(dofs_per_block);
  psi_old.reinit(dofs_per_block);
  rhs.reinit(dofs_per_block);

  // Initial condition
  Psi0<dim> ic;
  VectorTools::interpolate(dof_handler, ic, psi_old);
  constraints.distribute(psi_old);

  psi = psi_old;
}

template <int dim>
void SchrodingerTD<dim>::assemble_M_and_H()
{
  M = 0.0;
  H = 0.0;

  FEValues<dim> fev(fe, quadrature,
                    update_values | update_gradients |
                    update_quadrature_points | update_JxW_values);

  const FEValuesExtractors::Scalar a_fe(0);
  const FEValuesExtractors::Scalar b_fe(1);

  Potential<dim> V;

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q = quadrature.size();

  FullMatrix<double> cell_M(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_H(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fev.reinit(cell);
    cell_M = 0.0;
    cell_H = 0.0;

    for (unsigned int q=0; q<n_q; ++q)
    {
      const double Vq = V.value(fev.quadrature_point(q));

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        const double phi_ai = fev[a_fe].value(i,q);
        const double phi_bi = fev[b_fe].value(i,q);

        const Tensor<1,dim> grad_ai = fev[a_fe].gradient(i,q);
        const Tensor<1,dim> grad_bi = fev[b_fe].gradient(i,q);

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          const double phi_aj = fev[a_fe].value(j,q);
          const double phi_bj = fev[b_fe].value(j,q);

          const Tensor<1,dim> grad_aj = fev[a_fe].gradient(j,q);
          const Tensor<1,dim> grad_bj = fev[b_fe].gradient(j,q);

          // Mass matrix for each component (no coupling):
          cell_M(i,j) += (phi_ai*phi_aj + phi_bi*phi_bj) * fev.JxW(q);

          // Hamiltonian H = (1/2)(-Δ) + V:
          // Weak form gives (1/2)∫ grad(phi)·grad(trial) + ∫ V phi trial.
          cell_H(i,j) += 0.5 * (grad_ai*grad_aj + grad_bi*grad_bj) * fev.JxW(q);
          cell_H(i,j) += Vq * (phi_ai*phi_aj + phi_bi*phi_bj) * fev.JxW(q);
        }
      }
    }

    cell->get_dof_indices(local);
    constraints.distribute_local_to_global(cell_M, local, M);
    constraints.distribute_local_to_global(cell_H, local, H);
  }

  constraints.condense(M);
  constraints.condense(H);
}

template <int dim>
void SchrodingerTD<dim>::build_CN_matrices()
{
  // A and B are the Crank–Nicolson left/right matrices:
  //
  // A = [ M        + dt/2 H ]
  //     [ -dt/2 H      M   ]
  //
  // B = [ M        - dt/2 H ]
  //     [ +dt/2 H      M   ]
  //
  // with the understanding that H maps between a and b blocks in the coupling.
  //
  // Here, because we built M and H as block-diagonal (same operator on a and b),
  // we explicitly fill A,B blocks using those diagonal blocks.

  A = 0.0;
  B = 0.0;

  // Diagonal blocks: M
  A.block(0,0).copy_from(M.block(0,0));
  A.block(1,1).copy_from(M.block(1,1));
  B.block(0,0).copy_from(M.block(0,0));
  B.block(1,1).copy_from(M.block(1,1));

  // Off-diagonal blocks: ± dt/2 * H (using the corresponding operator)
  // We use H.block(0,0) as the operator on a-space, and H.block(1,1) on b-space.
  // Because spaces are isomorphic (same mesh/FE), these blocks have identical structure.
  A.block(0,1).copy_from(H.block(0,0));
  A.block(0,1) *= ( +0.5 * dt);

  A.block(1,0).copy_from(H.block(1,1));
  A.block(1,0) *= ( -0.5 * dt);

  B.block(0,1).copy_from(H.block(0,0));
  B.block(0,1) *= ( -0.5 * dt);

  B.block(1,0).copy_from(H.block(1,1));
  B.block(1,0) *= ( +0.5 * dt);

  // Constraints already condensed into M and H; A and B inherit consistent sparsity.
}

template <int dim>
void SchrodingerTD<dim>::assemble_rhs()
{
  rhs = 0.0;
  B.vmult(rhs, psi_old);
  constraints.condense(rhs);
}

template <int dim>
void SchrodingerTD<dim>::solve_step()
{
  // Solve A * psi = rhs using GMRES.
  SolverControl control(/*max_steps=*/6000, /*tol=*/1e-12 * rhs.l2_norm());
  SolverGMRES<BlockVector<double>> solver(control);

  PreconditionJacobi<BlockSparseMatrix<double>> pre;
  pre.initialize(A, 1.0);

  solver.solve(A, psi, rhs, pre);
  constraints.distribute(psi);
}

template <int dim>
void SchrodingerTD<dim>::output(unsigned int step) const
{
  // Since we used DoFRenumbering::component_wise(dof_handler),
  // all 'a' DoFs come first, then all 'b' DoFs. That means we can
  // safely pack the BlockVector into a monolithic Vector by concatenation.

  const auto n_a = dofs_per_block[0];
  const auto n_b = dofs_per_block[1];

  Vector<double> psi_mono(dof_handler.n_dofs());

  // Copy a-block
  for (types::global_dof_index i = 0; i < n_a; ++i)
    psi_mono[i] = psi.block(0)[i];

  // Copy b-block
  for (types::global_dof_index i = 0; i < n_b; ++i)
    psi_mono[n_a + i] = psi.block(1)[i];

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  // Avoid ambiguous overloads: pass vector<string>
  const std::vector<std::string> names = {"a", "b"};
  data_out.add_data_vector(psi_mono, names);

  data_out.build_patches();

  const std::string fn = "psi_" + Utilities::int_to_string(step, 4) + ".vtu";
  std::ofstream out(fn);
  data_out.write_vtu(out);

  std::cout << "Wrote " << fn << "\n";
}

template <int dim>
void SchrodingerTD<dim>::run()
{
  make_grid();
  setup_system();
  assemble_M_and_H();
  build_CN_matrices();

  unsigned int step = 0;
  double time = 0.0;

  output(step);

  const unsigned int n_steps = static_cast<unsigned int>(T / dt + 0.5);
  for (unsigned int n=0; n<n_steps; ++n)
  {
    time += dt;
    ++step;

    assemble_rhs();
    solve_step();

    psi_old = psi;

    if (output_every > 0 && (step % output_every == 0))
      output(step);
  }
}

// ---------------------- main ----------------------
int main()
{
  try
  {
    SchrodingerTD<2> p(1);
    p.run();
  }
  catch (std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

