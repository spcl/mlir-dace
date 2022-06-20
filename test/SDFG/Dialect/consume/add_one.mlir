// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>){
  // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
  // CHECK-SAME: !sdfg.stream<i32>
  %A = sdfg.alloc() : !sdfg.stream<i32>
  // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]] = sdfg.alloc
  // CHECK-SAME: !sdfg.array<6xi32>
  %C = sdfg.alloc() : !sdfg.array<6xi32>
  // CHECK: sdfg.state
  // CHECK-SAME: @state_0
  sdfg.state @state_0 {
    // CHECK: func.func @empty
    func.func @empty(%x: !sdfg.stream<i32>) -> i1{
      %0 = arith.constant 0 : i32
      %length = sdfg.stream_length %x : !sdfg.stream<i32> -> i32
      %isZero = arith.cmpi "eq", %length, %0 : i32
      return %isZero : i1
    }
    // CHECK: sdfg.consume
    // CHECK-DAG: num_pes = 5
    // CHECK-DAG: condition = @empty
    // CHECK-SAME: [[NAMEA]] : !sdfg.stream<i32>
    // CHECK-SAME: pe: [[NAMEp:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: elem: [[NAMEe:%[a-zA-Z0-9_]*]]
    sdfg.consume{num_pes=5, condition=@empty} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
      // CHECK: [[NAMEres:%[a-zA-Z0-9_]*]] = sdfg.tasklet
      // CHECK-SAME: [[NAMEe]]
      %res = sdfg.tasklet(%e: i32) -> (i32) {
          %1 = arith.constant 1 : i32
          %res = arith.addi %e, %1 : i32
          sdfg.return %res : i32
      }
      // CHECK: [[NAMEzero:%[a-zA-Z0-9_]*]] = sdfg.tasklet
      %0 = sdfg.tasklet() -> (index) {
          %0 = arith.constant 0 : index
          sdfg.return %0 : index
      }
      // CHECK: sdfg.store {wcr = "add"} [[NAMEres]], [[NAMEC]]
      // CHECK-SAME: [[NAMEzero]]
      sdfg.store{wcr="add"} %res, %C[%0] : i32 -> !sdfg.array<6xi32>
    }
  }
}
