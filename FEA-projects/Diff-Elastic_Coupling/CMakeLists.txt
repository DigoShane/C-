cmake_minimum_required(VERSION 3.13)
project(chemo_mechanical)

find_package(deal.II 9.3 REQUIRED HINTS /usr/local/lib/cmake/deal.II)

deal_ii_initialize_cached_variables()

add_executable(chemo_mechanical main.cc)

deal_ii_setup_target(chemo_mechanical)

