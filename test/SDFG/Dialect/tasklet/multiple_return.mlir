// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
  // CHECK: sdfg.state @state_0
  sdfg.state @state_0{
    // CHECK: sdfg.tasklet 
    // CHECK-SAME: i32
    %res:2 = sdfg.tasklet() -> (i32, i32) {
      // CHECK-NEXT: [[NAME0:%[a-zA-Z0-9_]*]]
      %0 = arith.constant 0 : i32
      // CHECK-NEXT: [[NAME1:%[a-zA-Z0-9_]*]]
      %1 = arith.constant 1 : i32
      // CHECK-NEXT: sdfg.return [[NAME0]], [[NAME1]]
      sdfg.return %0, %1 : i32, i32
    }

    // CHECK: sdfg.tasklet 
    sdfg.tasklet(%res#0:i32, %res#1:i32) -> (i32) {
      sdfg.return %res#1 : i32
    }
  }
}
