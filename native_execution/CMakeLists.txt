cmake_minimum_required(VERSION 3.27)

project(glm53_native_execution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build extensions as shared libraries" ON)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind 2.15.0 EXACT CONFIG REQUIRED)

# mlx.extension exports the wheel root through MLX_DIR.  Avoid `python -m mlx`
# here: that command initializes Metal and makes cross/headless compilation
# unnecessarily depend on a visible GPU.
list(APPEND CMAKE_PREFIX_PATH "$ENV{MLX_DIR}/share/cmake")
find_package(MLX CONFIG REQUIRED)

add_library(glm53_native_execution_core)
target_sources(
  glm53_native_execution_core
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/native_indexer_plan.cpp
  ${CMAKE_CURRENT_LIST_DIR}/native_indexpool_update_plan.cpp
  ${CMAKE_CURRENT_LIST_DIR}/native_dsa_score_plan.cpp
  ${CMAKE_CURRENT_LIST_DIR}/native_dsa_attention_plan.cpp)
target_include_directories(
  glm53_native_execution_core PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(glm53_native_execution_core PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET glm53_native_execution_metallib
    TITLE glm53_native_execution
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/native_indexer_plan.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(
    glm53_native_execution_core glm53_native_execution_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE glm53_native_execution_core)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
