// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc_stream()
    // CHECK-SAME: !sdir.stream_array<93x12xi32>
    %A = sdir.alloc_stream() : !sdir.stream_array<93x12xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.stream_array<93x12xi32> -> !sdir.stream<93x12xi32>
        %a = sdir.get_access %A : !sdir.stream_array<93x12xi32> -> !sdir.stream<93x12xi32>
    }
} 
