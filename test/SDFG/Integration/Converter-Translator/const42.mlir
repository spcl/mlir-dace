// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../import_translation_test.py
func.func private @main(%out: memref<i32>){
  %c42 = arith.constant 42 : i32
  memref.store %c42, %out[] : memref<i32>
  return
}