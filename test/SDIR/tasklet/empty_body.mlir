// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.tasklet 
// CHECK-SAME: @get_zero
sdir.tasklet @get_zero() -> i32{
}

