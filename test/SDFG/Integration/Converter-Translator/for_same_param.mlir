// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../import_translation_test.py
func.func private @main() {
  %c1 = arith.constant 1 : index
  scf.for %arg11 = %c1 to %c1 step %c1 {
    %c2 = arith.constant 1 : index
  }
  return
}
