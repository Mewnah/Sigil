# SPDX-License-Identifier: AGPL-3.0-only
#
# Sigil native engines: uniquely named sigil_<engine>.dll files implementing the sigil_engine ABI v1
# (common/sigil_engine.h), loaded lazily by Sigil.exe (src-tauri/src/services/native_runtime) from
# runtime packs. One build tree per variant (cpu, cpu-avx2, cpu-avx512, vulkan, cuda13):
#
#   pwsh scripts/native/build-runtime-packs.ps1 -Engine echo -Variant cpu      # build + package
#   cmake --preset cpu-avx2 && cmake --build --preset cpu-avx2                  # or by hand
#
# Build rules (enforced by cmake/SigilVariant.cmake and cmake/SigilEngine.cmake):
# - /MD (the shared UCRT / VC runtime) for every target, third-party code included;
# - third-party libraries (ggml, ...) are linked statically INTO each engine DLL (BUILD_SHARED_LIBS
#   OFF), with GGML_NATIVE=OFF, GGML_BACKEND_DL=OFF and the instruction sets of the variant, so two
#   engines never share a same-named ggml.dll and no build uses the build machine's CPU features;
# - every engine gets the shared ABI implementation (common/sigil_engine_impl.cpp) compiled in,
#   hidden symbols except the 16 exports, a version resource, CFG and a PDB.
#
# Engines live in engines/<name>/ (CMakeLists.txt + engine.json); each workstream owns its folder.
cmake_minimum_required(VERSION 3.25)
project(sigil_native VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Sigil engines are 64-bit only (configure with -A x64).")
endif()
if(NOT WIN32)
  message(WARNING "Sigil engines target Windows; other platforms are untested.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(SIGIL_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")

include(SigilVariant)
include(SigilEngine)

add_subdirectory(common)

set(SIGIL_ENGINES "" CACHE STRING "Engines to build (semicolon list of engines/<name>); empty = every engine")

file(GLOB _sigil_engine_dirs LIST_DIRECTORIES true RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/engines"
  "${CMAKE_CURRENT_SOURCE_DIR}/engines/*")
list(SORT _sigil_engine_dirs)
set(_sigil_found "")
foreach(_name IN LISTS _sigil_engine_dirs)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engines/${_name}" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/engines/${_name}/CMakeLists.txt")
    list(APPEND _sigil_found "${_name}")
    if(SIGIL_ENGINES STREQUAL "" OR _name IN_LIST SIGIL_ENGINES)
      message(STATUS "Sigil engine: ${_name} (${SIGIL_VARIANT})")
      add_subdirectory("engines/${_name}")
    endif()
  endif()
endforeach()
foreach(_wanted IN LISTS SIGIL_ENGINES)
  if(NOT _wanted IN_LIST _sigil_found)
    message(FATAL_ERROR "SIGIL_ENGINES names '${_wanted}', but engines/${_wanted}/CMakeLists.txt does not exist "
                        "(available: ${_sigil_found})")
  endif()
endforeach()
