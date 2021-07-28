// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.tasklet 
// CHECK-SAME: @get_zero
// CHECK-NEXT: [[NAME:%[a-zA-Z0-9_]*]]
// CHECK-NEXT: sdir.return [[NAME]]
sdir.tasklet @get_zero() -> i32{
    %c = constant 0 : i32
    sdir.return %c : i32
}

