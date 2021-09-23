// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.alloc_transient_stream()
        // CHECK-SAME: !sdir.stream_array<i32>
        %A = sdir.alloc_transient_stream() : !sdir.stream_array<i32>
    }
}

