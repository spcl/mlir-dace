configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SDIR_TEST_DEPENDS FileCheck count not sdir-opt)

add_lit_testsuite(
    check-sdir-converter "Running the sdir conversion tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${SDIR_TEST_DEPENDS}
)

set_target_properties(check-sdir-converter PROPERTIES FOLDER "Tests")
add_lit_testsuites(SDIR-TRANSLATE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SDIR_TEST_DEPENDS})
