#include "hyperelasticity_problem.h"
#include <iostream>

// Entry point. We simply instantiate the problem in 2D and run it.
int main()
{
  try
  {
    // degree=1 => Q1 elements (minimal, robust)
    dealii_hyper::HyperElasticityProblem<2> problem(/*degree=*/1);
    problem.run();
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "ERROR: Unknown exception\n";
    return 1;
  }
  return 0;
}

