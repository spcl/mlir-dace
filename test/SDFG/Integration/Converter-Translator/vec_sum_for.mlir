// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../execute_sdfg.py | FileCheck %s

// CHECK: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-COUNT-64: 3
// CHECK-NEXT: end_dump: [[ARRAY]]

// CHECK-NEXT: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 192
// CHECK-NEXT: end_dump: [[ARRAY]]

module {
  func.func @main(%A: memref<64xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c0_f64 = arith.constant 0.0 : f64
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    %sum = scf.for %k = %c0 to %c64 step %c1 iter_args(%accum = %c0_f64) -> (f64) {
      %a = memref.load %A[%k] : memref<64xf64>
      %r = arith.addf %accum, %a : f64
      scf.yield %r : f64
    }

    return %sum : f64
  }
}
