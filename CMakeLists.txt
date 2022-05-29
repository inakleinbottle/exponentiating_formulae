cmake_minimum_required(VERSION 3.19)
project(exponentiating_formulae)

set(CMAKE_CXX_STANDARD 14)
set(Boost_NO_WARN_NEW_VERSIONS 1)
add_subdirectory(libalgebra)

# include_directories(headers)
add_executable(
  exponentiating_formulae
  main.cpp
  "cbh_formulae.cpp"
  "environment.h"
  "headers/libalgebra/alternative_multiplications.h"
  "headers/libalgebra/area_tensor_basis.h"
  "headers/libalgebra/area_tensor_multiplication.h"
  "headers/libalgebra/half_shuffle_tensor_basis.h"
  "headers/libalgebra/half_shuffle_tensor_multiplication.h")
target_include_directories(exponentiating_formulae PRIVATE headers)
target_link_libraries(exponentiating_formulae PUBLIC Libalgebra::Libalgebra)
