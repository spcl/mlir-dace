// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.tasklet 
// CHECK-SAME: @get_zero
// CHECK-NEXT: [[NAME:%[a-zA-Z0-9_]*]]
sdir.tasklet @get_zero() -> i32{
    %c = constant 0 : i32
}

