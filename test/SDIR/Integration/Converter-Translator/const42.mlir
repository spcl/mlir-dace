// RUN: sdir-opt --convert-to-sdir %s | sdir-translate --mlir-to-sdfg | python %S/../import_translation_test.py
func private @const42(%out: memref<i32>){
  %c42 = arith.constant 42 : i32
  memref.store %c42, %out[] : memref<i32>
  return
}
