cmake_minimum_required(VERSION 3.16)
project(schrodinger_td LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(deal.II REQUIRED HINTS /usr/local)
deal_ii_initialize_cached_variables()

add_executable(schrodinger_td src/main.cc)
deal_ii_setup_target(schrodinger_td)

