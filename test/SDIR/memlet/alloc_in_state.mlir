// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: sdir.state
// CHECK-SAME: @state_0
// CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc()
// CHECK-SAME: !sdir.memlet<i32>
sdir.state @state_0 {
    %a = sdir.alloc() : !sdir.memlet<i32>
}


