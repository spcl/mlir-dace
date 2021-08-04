// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc_stream
%A = sdir.alloc_stream() : !sdir.stream<i32>

// CHECK: sdir.state
// CHECK-SAME: @state_0
sdir.state @state_0 {
    // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.stream_pop [[NAMEA]]
    // CHECK-SAME: !sdir.stream<i32>
    %a = sdir.stream_pop %A : !sdir.stream<i32>
}
