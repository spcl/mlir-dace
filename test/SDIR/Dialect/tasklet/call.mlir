// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state @state_0
    sdir.state @state_0{
        // CHECK: sdir.tasklet @one
        %c = sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }

        // CHECK: sdir.tasklet 
        // CHECK-SAME: @add
        // CHECK-SAME: [[NAMEA:%[a-zA-Z0-9_]*]]
        // CHECK-SAME: [[NAMEB:%[a-zA-Z0-9_]*]]
        %s = sdir.tasklet @add(%c as %org: i32) -> i32{
            // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
            // CHECK-SAME: [[NAMEA]] 
            // CHECK-SAME: [[NAMEB]]
            %r = arith.addi %org, %org : i32
            // CHECK-NEXT: sdir.return [[NAMEC]]
            sdir.return %r : i32
        }

    }
}
