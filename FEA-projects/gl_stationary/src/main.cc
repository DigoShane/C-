#include "gl_problem.h"
#include <iostream>

int main()
{
  try
  {
    // degree=1 is a good minimalist starting point
    gl::GLProblem<2> problem(/*degree=*/1);

    // You can tune these:
    problem.set_kappa(2.0);  // GL parameter
    problem.set_H(8.0);      // applied field strength in symmetric gauge

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

