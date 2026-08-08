cmake_minimum_required(VERSION 3.20)
project(liblevenshtein_package_smoke LANGUAGES CXX)

find_package(liblevenshtein 0.10 CONFIG REQUIRED)

add_executable(liblevenshtein_package_smoke ../snapshot.cpp)
target_compile_features(liblevenshtein_package_smoke PRIVATE cxx_std_20)
target_link_libraries(liblevenshtein_package_smoke PRIVATE liblevenshtein::liblevenshtein)
