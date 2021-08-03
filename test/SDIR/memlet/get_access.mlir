// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// Module is always implicitly top-level
// CHECK: module
// CHECK-SAME: @state_0
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEB:%[a-zA-Z0-9_]*]]
// CHECK-SAME: !sdir.memlet<i32>
sdir.state @state_0 {
    %a = sdir.get_access : !sdir.memlet<i32>
}
