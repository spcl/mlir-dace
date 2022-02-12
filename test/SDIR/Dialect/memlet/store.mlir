// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<i32>
    %A = sdir.alloc() : !sdir.array<i32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.tasklet @one
        %1 = sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        // CHECK: sdir.store [[NAMEC]], [[NAMEA]][]
        // CHECK-SAME: i32 -> !sdir.array<i32>
        sdir.store %1, %A[] : i32 -> !sdir.array<i32>
    }
}
