// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-translate --mlir-to-sdfg | python3 %S/../execute_sdfg.py | FileCheck %s

// CHECK: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-COUNT-64: 3
// CHECK-NEXT: end_dump: [[ARRAY]]

// CHECK-NEXT: begin_dump: [[ARRAY:[a-zA-Z0-9_]*]]
// CHECK-NEXT: 192
// CHECK-NEXT: end_dump: [[ARRAY]]

module {
  func.func @main(%arg0: memref<64xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0:2 = scf.while (%arg1 = %c0, %arg2 = %cst) : (index, f64) -> (index, f64) {
      %1 = arith.cmpi slt, %arg1, %c64 : index
      scf.condition(%1) %arg1, %arg2 : index, f64
    } do {
    ^bb0(%arg1: index, %arg2: f64):
      %1 = arith.addi %arg1, %c1 : index
      %2 = memref.load %arg0[%arg1] : memref<64xf64>
      %3 = arith.addf %arg2, %2 : f64
      scf.yield %1, %3 : index, f64
    }
    return %0#1 : f64
  }
}
