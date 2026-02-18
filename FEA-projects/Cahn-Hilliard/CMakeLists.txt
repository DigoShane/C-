cmake_minimum_required(VERSION 3.10)

project(CahnHilliard)

# -------------------------------------------------------
# Locate deal.II
# -------------------------------------------------------

find_package(deal.II 9.3 REQUIRED
  HINTS ${DEAL_II_DIR}
)

if(NOT deal.II_FOUND)
  message(FATAL_ERROR "deal.II not found. Set DEAL_II_DIR.")
endif()

deal_ii_initialize_cached_variables()

# -------------------------------------------------------
# Build executable
# -------------------------------------------------------

add_executable(cahn_hilliard main.cc)

deal_ii_setup_target(cahn_hilliard)

