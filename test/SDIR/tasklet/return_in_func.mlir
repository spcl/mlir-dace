// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: func
// CHECK-SAME: @add
// CHECK-SAME: [[NAMEA:%[a-zA-Z0-9_]*]]
// CHECK-SAME: [[NAMEB:%[a-zA-Z0-9_]*]]
func @add(%a: i32, %b: i32) -> i32{
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[NAMEA]] 
    // CHECK-SAME: [[NAMEB]]
    %c = addi %a, %b : i32
    // CHECK-NEXT: sdir.return [[NAMEC]]
    sdir.return %c : i32
}