// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc
%A = sdir.alloc() : !sdir.memlet<i32>

// CHECK: sdir.state
// CHECK-SAME: @state_0
sdir.state @state_0 {
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
    // CHECK-SAME: !sdir.memlet<i32>
    %a = sdir.get_access %A : !sdir.memlet<i32>
    // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.load [[NAMEB]][0]
    // CHECK-SAME: !sdir.memlet<i32>
    %a_1 = sdir.load %a[0] : !sdir.memlet<i32>
}
