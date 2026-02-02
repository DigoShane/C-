cmake_minimum_required(VERSION 3.16)
project(hyperelasticity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(deal.II REQUIRED HINTS /usr/local)
deal_ii_initialize_cached_variables()

add_executable(hyperelastic
  src/main.cc
  src/hyperelasticity_problem.cc
)

target_include_directories(hyperelastic PRIVATE src)
deal_ii_setup_target(hyperelastic)

