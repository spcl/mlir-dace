configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SDIR_TEST_DEPENDS FileCheck count not sdir-opt sdir-translate)

add_lit_testsuite(
    check-sdir-integration "Running the sdir integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${SDIR_TEST_DEPENDS}
)

set_target_properties(check-sdir-integration PROPERTIES FOLDER "Tests")
add_lit_testsuites(SDIR-INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SDIR_TEST_DEPENDS})
