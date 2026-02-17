cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.7.1 REQUIRED
  HINTS ${DEAL_II_DIR}
)

deal_ii_initialize_cached_variables()

project(project1)

add_executable(project1 main.cc)

deal_ii_setup_target(project1)
