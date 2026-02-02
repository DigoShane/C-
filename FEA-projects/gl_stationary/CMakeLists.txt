cmake_minimum_required(VERSION 3.16)
project(gl_stationary LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(deal.II REQUIRED HINTS /usr/local)
deal_ii_initialize_cached_variables()

add_executable(gl_stationary
  src/main.cc
  src/gl_problem.cc
)

target_include_directories(gl_stationary PRIVATE src)
deal_ii_setup_target(gl_stationary)

