cmake_minimum_required(VERSION 3.22)
project(exponentiating_formulae)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(libalgebra)


add_executable(exponentiating_formulae main.cpp)
target_link_libraries(exponentiating_formulae PUBLIC Libalgebra::Libalgebra)
