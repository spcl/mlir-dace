// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state @state_0
    sdir.state @state_0{
        // CHECK: sdir.tasklet 
        // CHECK-SAME: @add
        // CHECK-SAME: [[NAMEA:%[a-zA-Z0-9_]*]]
        // CHECK-SAME: [[NAMEB:%[a-zA-Z0-9_]*]]
        %res = sdir.tasklet @add(%a: i32, %b: i32) -> i32 {
            // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
            // CHECK-SAME: [[NAMEA]] 
            // CHECK-SAME: [[NAMEB]]
            %c = arith.addi %a, %b : i32
            // CHECK-NEXT: sdir.return [[NAMEC]]
            sdir.return %c : i32
        }
    }
}
