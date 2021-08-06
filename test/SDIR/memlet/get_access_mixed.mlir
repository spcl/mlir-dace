// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc_stream()
// CHECK-SAME: !sdir.stream_array<i32>
%A = sdir.alloc_stream() : !sdir.stream_array<i32>

// CHECK: sdir.state
// CHECK-SAME: @state_0
// CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
// CHECK-SAME: !sdir.stream_array<i32> -> !sdir.memlet<i32>
sdir.state @state_0 {
    %a = sdir.get_access %A : !sdir.stream_array<i32> -> !sdir.memlet<i32>
}
