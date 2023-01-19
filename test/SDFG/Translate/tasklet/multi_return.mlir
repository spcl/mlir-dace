// XFAIL: *
// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py
// TODO: Implement test with tasklet returning multiple values
