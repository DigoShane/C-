// ============================================================================
// 2D Linear Elasticity Solver using deal.II
//
// Problem:
//   - Small-strain linear elasticity (plane strain)
//   - Rectangular domain [0,L] x [0,H]
//   - Left boundary clamped (u = 0)
//   - Right boundary subjected to constant traction
//
// Purpose:
//   - Minimal but real FEM simulation
//   - Demonstrates mesh setup, DoF handling, assembly, solver, output
//   - Clean C++ structure suitable for CV / interviews
// ============================================================================

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// ============================================================================
// Traction function
//
// This defines the Neumann boundary condition on the right boundary (x = L).
// We apply a constant traction vector t = (T, 0).
// ============================================================================

template <int dim>
class RightTraction : public TensorFunction<1, dim>
{
public:
  explicit RightTraction(const double T)
    : TensorFunction<1, dim>(), T_(T)
  {}

  // Value of traction at a boundary point
  virtual Tensor<1, dim> value(const Point<dim> &) const override
  {
    Tensor<1, dim> t;
    t[0] = T_;      // traction in x-direction
    t[1] = 0.0;     // no vertical traction
    return t;
  }

private:
  const double T_;
};

// ============================================================================
// ElasticityProblem class
//
// <dim> - spatial dimension (2D). Defined with class template.
//
// This class encapsulates the entire FEM pipeline:
//   - mesh generation
//   - DoF setup and constraints
//   - system assembly
//   - linear solve
//   - output
// ============================================================================

template <int dim>
class ElasticityProblem
{
public:
  explicit ElasticityProblem(const unsigned int degree);
  void run();

private:
  // High-level FEM steps
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const std::string &filename) const;

  // Helper: assign boundary IDs based on geometry
  void set_boundary_ids();

  // Core deal.II objects
  Triangulation<dim> triangulation; //stores the mesh (cells faces connectvity).
  FESystem<dim>      fe;           // Function space. FESystem<dim>(FE_Q<dim>(degree), dim)
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;//For dirichlet BC.

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix; // stokes K

  Vector<double> solution; //stores u
  Vector<double> system_rhs; //stores F

  // Material parameters (Lamé constants)
  double lambda;
  double mu;

  // Traction magnitude
  double traction_T;

  // Geometry parameters
  double L;
  double H;
};

// ============================================================================
// Constructor
//
// Sets geometry size, material parameters, and traction magnitude.
// ============================================================================

template <int dim>
ElasticityProblem<dim>::ElasticityProblem(const unsigned int degree)
  : fe(FE_Q<dim>(degree), dim)   // FE_Q for each displacement component
  , dof_handler(triangulation)
{
  // Geometry
  L = 1.0;
  H = 0.2;

  // Linear elastic material (plane strain)
  const double E  = 1.0;   // Young's modulus (scaled)
  const double nu = 0.3;   // Poisson ratio

  mu     = E / (2.0 * (1.0 + nu));
  lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));

  // Applied traction
  traction_T = 1e-2;
}

// ============================================================================
// Mesh generation
//
// Creates a rectangular grid and assigns boundary IDs.
// ============================================================================

template <int dim>
void ElasticityProblem<dim>::make_grid()
{
  const unsigned int Nx = 40;
  const unsigned int Ny = 8;

  GridGenerator::subdivided_hyper_rectangle(
    triangulation,
    {Nx, Ny},
    Point<dim>(0.0, 0.0),
    Point<dim>(L, H),
    /*colorize*/ true);

  set_boundary_ids();

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells() << "\n";
}

// ============================================================================
// Boundary ID assignment
//
// We explicitly label:
//   - x = 0  -> boundary id 1 (clamped)
//   - x = L  -> boundary id 2 (traction)
// ============================================================================

template <int dim>
void ElasticityProblem<dim>::set_boundary_ids()
{
  const double tol = 1e-12;

  for (const auto &cell : triangulation.active_cell_iterators()) // loops over active cells.
    for (const auto &face : cell->face_iterators()) // iterates over faces of a cell.
      if (face->at_boundary())
      {
        const auto c = face->center();
        if (std::abs(c[0]) < tol)
          face->set_boundary_id(1);
        else if (std::abs(c[0] - L) < tol)
          face->set_boundary_id(2);
      }
}

// ============================================================================
// DoF setup and constraints
//
// - Distribute degrees of freedom
// - Apply Dirichlet BC on boundary id 1
// - Build sparsity pattern and vectors
// ============================================================================

template <int dim>
void ElasticityProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs() << "\n";

  constraints.clear();

  // Clamp left boundary (u = 0) on bdry id 1.
  VectorTools::interpolate_boundary_values(
    dof_handler,
    1,
    Functions::ZeroFunction<dim>(dim),
    constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

// ============================================================================
// Assembly
//
// Assembles:
//   - Volume stiffness matrix (linear elasticity)
//   - Neumann traction contribution on right boundary
// ============================================================================

template <int dim>
void ElasticityProblem<dim>::assemble_system()
{
  system_matrix = 0.0;
  system_rhs    = 0.0;

  const QGauss<dim>     quadrature(fe.degree + 1);
  const QGauss<dim-1>   face_quadrature(fe.degree + 1);

  FEValues<dim> fe_values(
    fe, quadrature,
    update_values | update_gradients | update_JxW_values);

  FEFaceValues<dim> fe_face_values(
    fe, face_quadrature,
    update_values | update_JxW_values | update_quadrature_points);

  const FEValuesExtractors::Vector u(0);
  RightTraction<dim> traction(traction_T);

  FullMatrix<double> cell_matrix(fe.n_dofs_per_cell());
  Vector<double>     cell_rhs(fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_matrix = 0.0;
    cell_rhs    = 0.0;
    fe_values.reinit(cell);

    // Volume term
    for (unsigned int q = 0; q < quadrature.size(); ++q)
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
      {
        const auto eps_i =
          symmetrize(fe_values[u].gradient(i, q));

        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
        {
          const auto eps_j =
            symmetrize(fe_values[u].gradient(j, q));

          const double stiffness =
            2.0 * mu * (eps_i * eps_j)
            + lambda * trace(eps_i) * trace(eps_j);

          cell_matrix(i, j) += stiffness * fe_values.JxW(q);
        }
      }

    // Traction boundary term
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      if (cell->face(face)->at_boundary() &&
          cell->face(face)->boundary_id() == 2)
      {
        fe_face_values.reinit(cell, face);
        for (unsigned int q = 0; q < face_quadrature.size(); ++q)
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            cell_rhs(i) +=
              (fe_face_values[u].value(i, q)
               * traction.value(fe_face_values.quadrature_point(q)))
              * fe_face_values.JxW(q);
      }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
      cell_matrix, cell_rhs,
      local_dof_indices,
      system_matrix, system_rhs);
  }
}

// ============================================================================
// Linear solve
// ============================================================================

template <int dim>
void ElasticityProblem<dim>::solve()
{
  SolverControl control(2000, 1e-12 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);

  std::cout << "CG iterations: " << control.last_step() << "\n";
}

// ============================================================================
// Output
// ============================================================================

template <int dim>
void ElasticityProblem<dim>::output_results(const std::string &filename) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  const std::vector<std::string> solution_names = {"u_x", "u_y"};
  data_out.add_data_vector(solution, solution_names);
  data_out.build_patches();

  std::ofstream out(filename);
  data_out.write_vtu(out);

  std::cout << "Wrote: " << filename << "\n";
}

// ============================================================================
// Run sequence
// ============================================================================

template <int dim>
void ElasticityProblem<dim>::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results("solution.vtu");
}

// ============================================================================
// main()
// ============================================================================

int main()
{
  try
  {
    ElasticityProblem<2> problem(1); // Q1 elements
    problem.run();
  }
  catch (std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

