# SPDX-License-Identifier: AGPL-3.0-only
#
# sigil::engine_abi - the shared sigil_engine ABI v1 implementation, compiled into every engine DLL
# (INTERFACE sources: a static library would drop the unreferenced export objects).
add_library(sigil_engine_abi INTERFACE)
add_library(sigil::engine_abi ALIAS sigil_engine_abi)
target_include_directories(sigil_engine_abi INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_sources(sigil_engine_abi INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/sigil_engine_impl.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/sigil_json.cpp")
target_compile_definitions(sigil_engine_abi INTERFACE SIGIL_ENGINE_BUILD)
target_compile_features(sigil_engine_abi INTERFACE cxx_std_17)

# sigil_engine_check.exe - loads a built engine like Sigil does and prints its description as JSON
# (the build script writes sigil-pack.json from it).
if(WIN32)
  add_executable(sigil_engine_check sigil_engine_check.cpp sigil_json.cpp)
  target_include_directories(sigil_engine_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(sigil_engine_check PRIVATE cxx_std_17)
  if(MSVC)
    target_compile_options(sigil_engine_check PRIVATE /W4 /permissive- /EHsc)
  endif()
  # The checker must run on any x64 build PC, whatever variant tree it is built in (it detects the
  # CPU and skips create for a variant the PC cannot execute): drop the variant's /arch flag, which
  # SigilVariant.cmake adds directory-wide.
  get_target_property(_sigil_check_opts sigil_engine_check COMPILE_OPTIONS)
  if(_sigil_check_opts)
    list(FILTER _sigil_check_opts EXCLUDE REGEX "/arch:")
    set_property(TARGET sigil_engine_check PROPERTY COMPILE_OPTIONS "${_sigil_check_opts}")
  endif()
  install(TARGETS sigil_engine_check RUNTIME DESTINATION tools)
  sigil_record_output(sigil_engine_check)
endif()
