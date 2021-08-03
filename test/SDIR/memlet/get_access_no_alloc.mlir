// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.state
// CHECK-SAME: @state_0
// CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
// CHECK-SAME: !sdir.memlet<i32>
sdir.state @state_0 {
    %a = sdir.get_access %A : !sdir.memlet<i32>
}
