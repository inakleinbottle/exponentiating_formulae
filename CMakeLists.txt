cmake_minimum_required(VERSION 3.19)
project(exponentiating_formulae)

set(CMAKE_CXX_STANDARD 17)
set(Boost_NO_WARN_NEW_VERSIONS 1)
add_subdirectory(libalgebra)

# include_directories(headers)
add_executable(
  exponentiating_formulae
  main.cpp
  "cbh_formulae.cpp"
  "environment.h"
  "danyu.cpp")
#target_include_directories(exponentiating_formulae PRIVATE headers)
target_link_libraries(exponentiating_formulae PUBLIC Libalgebra::Libalgebra)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT exponentiating_formulae)
