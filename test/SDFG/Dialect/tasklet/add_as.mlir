// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK: sdfg.state @state_0
    sdfg.state @state_0{
        // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
        // CHECK-SAME: sdfg.tasklet @one
        %c = sdfg.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdfg.return %1 : i32
        }

        // CHECK: sdfg.tasklet @add
        // CHECK-SAME: [[NAMEC]] as [[NAMEB:%[a-zA-Z0-9_]*]]
        %s = sdfg.tasklet @add(%c as %b: i32) -> i32{
            // CHECK-NEXT: [[NAMER:%[a-zA-Z0-9_]*]]
            // CHECK-SAME: [[NAMEB]], [[NAMEB]]
            %r = arith.addi %b, %b : i32
            // CHECK-NEXT: sdfg.return [[NAMER]]
            sdfg.return %r : i32
        }

    }
}
