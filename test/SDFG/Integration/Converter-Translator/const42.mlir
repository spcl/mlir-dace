// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../execute_sdfg.py | FileCheck %s
// CHECK: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 42
// CHECK-NEXT: end_dump: [[ARRAY]]
func.func private @main() -> i32{
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}
