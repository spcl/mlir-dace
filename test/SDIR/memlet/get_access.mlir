// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
%A = sdir.alloc() : !sdir.array<i32>

// CHECK: sdir.state
// CHECK-SAME: @state_0
// CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
// CHECK-SAME: !sdir.array<i32> -> !sdir.memlet<i32>
sdir.state @state_0 {
    %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
}
