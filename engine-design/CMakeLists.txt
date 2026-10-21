cmake_minimum_required(VERSION 3.20)
project(dgemma_design_checks LANGUAGES C CXX)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(dg-contract-c tests/header-smoke.c)
add_executable(dg-contract-cpp tests/header-smoke.cpp)
set_target_properties(dg-contract-c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
set_target_properties(dg-contract-cpp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
foreach(target dg-contract-c dg-contract-cpp)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
endforeach()
add_test(NAME design-header-c COMMAND dg-contract-c)
add_test(NAME design-header-cpp COMMAND dg-contract-cpp)
add_test(
    NAME design-workflow-policy
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_workflow.py -v
)

# I0's exact package oracle lives only in the ignored build tree.  A missing
# interpreter is an explicit BLOCKED prerequisite: CTest fails to launch the
# mandatory groups rather than reporting a skipped pass.  Callers may point
# this cache variable at another isolated environment with the same lock.
set(
    DG_ENGINE_ORACLE_PYTHON
    "${CMAKE_CURRENT_SOURCE_DIR}/../build/i0-oracle-venv/bin/python"
    CACHE FILEPATH
    "Python with exactly transformers 5.13.0 and diffusers 0.39.0"
)
set(
    DG_ENGINE_I0_EVIDENCE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../build/i0-evidence-source"
    CACHE PATH
    "External I0 evidence root containing i0-evidence.json and relative artifacts"
)
add_test(
    NAME BASE-01
    COMMAND ${DG_ENGINE_ORACLE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_i0_group.py
            BASE-01 --evidence-root ${DG_ENGINE_I0_EVIDENCE_ROOT}
)
add_test(
    NAME BASE-02
    COMMAND ${DG_ENGINE_ORACLE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_i0_group.py
            BASE-02 --evidence-root ${DG_ENGINE_I0_EVIDENCE_ROOT}
)
# run_i0_group.py verifies collection and execution counts and rejects skips,
# failures, errors, expected failures, and unavailable mandatory evidence.  The
# BASE-01's short timeout is a fail-closed anti-hang guard for FIFO/special-file
# security regressions.  BASE-02 may stream-hash roughly 146 GB of checkpoint
# and GGUF gate evidence, so its fail-closed limit allows realistic slow/network
# evidence stores instead of misreporting valid I/O as a semantic failure.
set_tests_properties(BASE-01 PROPERTIES LABELS "I0" TIMEOUT 180)
set_tests_properties(BASE-02 PROPERTIES LABELS "I0" TIMEOUT 3600)
