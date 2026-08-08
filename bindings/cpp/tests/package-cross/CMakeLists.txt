cmake_minimum_required(VERSION 3.20)
project(vinary_tree_cross_project_package_smoke LANGUAGES CXX)

find_package(libdictenstein 0.2 CONFIG REQUIRED)
find_package(liblevenshtein 0.10 CONFIG REQUIRED)

add_executable(
  vinary_tree_cross_project_package_smoke
  ../../../../../libdictenstein/bindings/cpp/tests/cross_project_snapshot.cpp
)
target_compile_features(vinary_tree_cross_project_package_smoke PRIVATE cxx_std_20)
target_link_libraries(
  vinary_tree_cross_project_package_smoke
  PRIVATE
    libdictenstein::libdictenstein
    liblevenshtein::liblevenshtein
)
