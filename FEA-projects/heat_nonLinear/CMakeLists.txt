cmake_minimum_required(VERSION 3.16)
project(heat_nonlinear LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(deal.II REQUIRED HINTS /usr/local)
deal_ii_initialize_cached_variables()

add_executable(heat_nonlinear src/main.cc)
deal_ii_setup_target(heat_nonlinear)

