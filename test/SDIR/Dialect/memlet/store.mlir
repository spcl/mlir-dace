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
        // CHECK: sdir.tasklet @one
        sdir.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdir.return %1 : i32
        }
        // CHECK: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.call @one()
        %1 = sdir.call @one() : () -> i32
        // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<i32> -> !sdir.memlet<i32>
        %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
        // CHECK-NEXT: sdir.store [[NAMEC]], [[NAMEB]][]
        // CHECK-SAME: i32 -> !sdir.memlet<i32>
        sdir.store %1, %a[] : i32 -> !sdir.memlet<i32>
    }
}
