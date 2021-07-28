// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.tasklet 
// CHECK-SAME: @add
// CHECK-SAME: [[NAMEA:%[a-zA-Z0-9_]*]]
// CHECK-SAME: [[NAMEB:%[a-zA-Z0-9_]*]]
// CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
// CHECK-SAME: [[NAMEA]] 
// CHECK-SAME: [[NAMEB]]
// CHECK-NEXT: sdir.return [[NAMEC]]
sdir.tasklet @add(%a: i32, %b: i32) -> i32{
    %c = addi %a, %b : i32
    sdir.return %c : i32
}