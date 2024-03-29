// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
  // CHECK-SAME: !sdfg.stream<2x6xi32>
  %A = sdfg.alloc() : !sdfg.stream<2x6xi32>
  // CHECK: sdfg.state
  // CHECK-SAME: @state_0
  sdfg.state @state_0 {
    // CHECK: func.func @empty
    func.func @empty(%x: !sdfg.stream<2x6xi32>) -> i1{
      %0 = arith.constant 0 : i32
      %length = sdfg.stream_length %x : !sdfg.stream<2x6xi32> -> i32
      %isZero = arith.cmpi "eq", %length, %0 : i32
      return %isZero : i1
    }
    // CHECK: sdfg.consume
    sdfg.consume{condition=@empty} (%A : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
    }
  }
}
