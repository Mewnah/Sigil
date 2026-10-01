# sigil-bergamot.exe: Mozilla's Bergamot translation engine behind Sigil's JSON-lines protocol
# (docs/adr/0008-bergamot.md). Built by scripts/runtime-packs/bergamot/build.ps1; see README.md.
#
#   Real engine:  -DBERGAMOT_SOURCE_DIR=<pinned mozilla/translations>/inference  (UPSTREAM.json)
#   Fake engine:  -DSIGIL_BERGAMOT_FAKE=ON  (protocol shell only, no upstream sources; for tests)
#
# Part of Sigil (AGPL-3.0).
cmake_minimum_required(VERSION 3.21)
project(sigil_bergamot CXX C)

option(SIGIL_BERGAMOT_FAKE "Build the protocol shell with the fake engine (no upstream sources)" OFF)
set(BERGAMOT_SOURCE_DIR "" CACHE PATH "The inference/ folder of the pinned mozilla/translations source tree")
set(SIGIL_BERGAMOT_VERSION "dev" CACHE STRING "Pack version (UPSTREAM.json catalog_version)")
set(SIGIL_BERGAMOT_UPSTREAM "" CACHE STRING "Upstream description, e.g. mozilla/translations 69455ac")
set(SIGIL_BERGAMOT_ARCH "avx2" CACHE STRING "Instruction set: avx2 (default), sse2 (compatibility) or avx512")
set_property(CACHE SIGIL_BERGAMOT_ARCH PROPERTY STRINGS avx2 sse2 avx512)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT, like the upstream MSVC build (/MT): the helper is one self-contained executable.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(SIGIL_SOURCES src/main.cpp src/json.cpp)

if(SIGIL_BERGAMOT_FAKE)
  set(SIGIL_BERGAMOT_ARCH_LABEL "fake")
  add_executable(sigil-bergamot ${SIGIL_SOURCES} src/engine_fake.cpp)
else()
  if(NOT EXISTS "${BERGAMOT_SOURCE_DIR}/CMakeLists.txt" OR NOT EXISTS "${BERGAMOT_SOURCE_DIR}/marian-fork")
    message(FATAL_ERROR "BERGAMOT_SOURCE_DIR must point at the inference/ folder of the pinned mozilla/translations tree "
                        "(fetched by scripts/runtime-packs/bergamot/build.ps1); got '${BERGAMOT_SOURCE_DIR}'")
  endif()
  set(SIGIL_BERGAMOT_ARCH_LABEL "${SIGIL_BERGAMOT_ARCH}")

  # Upstream's own options for a native MSVC build without MKL/BLAS: the WebAssembly-compatible
  # sources (intgemm for the int8 models, the ONNX.js SGEMM for float math), library only, static
  # third-party code, no tests, no Python bindings, no GPU.
  set(USE_WASM_COMPATIBLE_SOURCE ON CACHE BOOL "" FORCE)
  set(USE_ONNX_SGEMM ON CACHE BOOL "" FORCE)
  set(USE_MKL OFF CACHE BOOL "" FORCE)
  set(COMPILE_CUDA OFF CACHE BOOL "" FORCE)
  set(COMPILE_TESTS OFF CACHE BOOL "" FORCE)
  set(COMPILE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(COMPILE_PYTHON OFF CACHE BOOL "" FORCE)
  set(COMPILE_LIBRARY_ONLY ON CACHE BOOL "" FORCE)
  set(USE_STATIC_LIBS ON CACHE BOOL "" FORCE)
  set(GIT_SUBMODULE OFF CACHE BOOL "" FORCE)
  # BUILD_ARCH drives upstream's MSVC /arch mapping (inference/CMakeLists.txt): x86-64-v3 -> /arch:AVX2,
  # x86-64-v2 -> /arch:SSE2 (intgemm still picks SSSE3/AVX2/AVX-512 kernels at run time),
  # x86-64-v4 -> /arch:AVX512. Never "native": friends' CPUs differ from the build PC.
  if(SIGIL_BERGAMOT_ARCH STREQUAL "avx2")
    set(BUILD_ARCH "x86-64-v3" CACHE STRING "" FORCE)
  elseif(SIGIL_BERGAMOT_ARCH STREQUAL "sse2")
    set(BUILD_ARCH "x86-64-v2" CACHE STRING "" FORCE)
  elseif(SIGIL_BERGAMOT_ARCH STREQUAL "avx512")
    set(BUILD_ARCH "x86-64-v4" CACHE STRING "" FORCE)
  else()
    message(FATAL_ERROR "SIGIL_BERGAMOT_ARCH must be avx2, sse2 or avx512")
  endif()

  add_subdirectory("${BERGAMOT_SOURCE_DIR}" bergamot EXCLUDE_FROM_ALL)
  if(NOT TARGET bergamot-translator)
    message(FATAL_ERROR "the upstream tree did not define the bergamot-translator library target")
  endif()

  add_executable(sigil-bergamot ${SIGIL_SOURCES} src/engine_bergamot.cpp)
  target_link_libraries(sigil-bergamot PRIVATE bergamot-translator)
  if(MSVC)
    # Match upstream's whole-program optimization and static CRT link settings.
    target_link_options(sigil-bergamot PRIVATE /LTCG /NODEFAULTLIB:MSVCRT /ignore:4049)
  endif()
endif()

if(MSVC)
  # /W3 and permissive mode: engine_bergamot.cpp includes upstream headers written for them.
  target_compile_options(sigil-bergamot PRIVATE /W3 /utf-8 /EHsc)
  if(SIGIL_BERGAMOT_ARCH_LABEL STREQUAL "avx2")
    target_compile_options(sigil-bergamot PRIVATE /arch:AVX2)
  elseif(SIGIL_BERGAMOT_ARCH_LABEL STREQUAL "avx512")
    target_compile_options(sigil-bergamot PRIVATE /arch:AVX512)
  endif()
  target_compile_definitions(sigil-bergamot PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_compile_definitions(sigil-bergamot PRIVATE
  SIGIL_BERGAMOT_VERSION="${SIGIL_BERGAMOT_VERSION}"
  SIGIL_BERGAMOT_ARCH="${SIGIL_BERGAMOT_ARCH_LABEL}")
if(SIGIL_BERGAMOT_UPSTREAM)
  target_compile_definitions(sigil-bergamot PRIVATE SIGIL_BERGAMOT_UPSTREAM="${SIGIL_BERGAMOT_UPSTREAM}")
endif()

install(TARGETS sigil-bergamot RUNTIME DESTINATION .)

# Unit tests of the JSON code (no upstream code; build.ps1 -Fake builds and runs them).
add_executable(sigil-bergamot-json-test EXCLUDE_FROM_ALL src/json_test.cpp src/json.cpp)
if(MSVC)
  target_compile_options(sigil-bergamot-json-test PRIVATE /W4 /utf-8 /EHsc /permissive-)
endif()
