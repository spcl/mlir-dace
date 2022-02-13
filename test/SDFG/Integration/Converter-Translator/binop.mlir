// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python %S/../import_translation_test.py
func private @binop(%arg1: i32, %arg2: i32, %out: memref<i32>) {
  %c0 = arith.addi %arg1, %arg2 : i32
  memref.store %c0, %out[] : memref<i32>
  return
}
