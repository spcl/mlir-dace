# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(SDFG_TEST_DEPENDS FileCheck count not sdfg-opt)

add_lit_testsuite(check-sdfg-opt "Running the sdfg regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${SDFG_TEST_DEPENDS})

set_target_properties(check-sdfg-opt PROPERTIES FOLDER "Tests")
add_lit_testsuites(SDFG-OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${SDFG_TEST_DEPENDS})
