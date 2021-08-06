// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg @sdfg_0 {
    // CHECK: sdir.state @state_0
    sdir.state @state_0{
        // CHECK: sdir.tasklet 
        // CHECK-SAME: @get_zero
        sdir.tasklet @get_zero() -> i32{
            // CHECK-NEXT: [[NAME:%[a-zA-Z0-9_]*]]
            %c = constant 0 : i32
        }
    }
}