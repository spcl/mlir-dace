// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state @state_0
    sdir.state @state_0{
        // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
        // CHECK-SAME: sdir.tasklet @one
        %c = sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }

        // CHECK: sdir.tasklet @add
        // CHECK-SAME: [[NAMEC]] as [[NAMEB1:%[a-zA-Z0-9_]*]]
        // CHECK-SAME: [[NAMEC]] as [[NAMEB2:%[a-zA-Z0-9_]*]]
        %s = sdir.tasklet @add(%c as %b1: i32, %c as %b2: i32) -> i32{
            // CHECK-NEXT: [[NAMER:%[a-zA-Z0-9_]*]]
            // CHECK-SAME: [[NAMEB1]], [[NAMEB2]]
            %r = arith.addi %b1, %b2 : i32
            // CHECK-NEXT: sdir.return [[NAMER]]
            sdir.return %r : i32
        }

    }
}
