// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state @state_0
    sdir.state @state_0{
        // CHECK: sdir.tasklet @one
        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        // CHECK: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.call @one()
        %1 = sdir.call @one() : () -> i32
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.call @add
        // CHECK-SAME: ([[NAME1]], [[NAME1]])
        // CHECK-SAME: (i32, i32) -> i32
        %c = sdir.call @add(%1, %1) : (i32, i32) -> i32
    }
}
