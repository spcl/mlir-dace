// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc_stream()
    // CHECK-SAME: !sdir.stream_array<2x6xi32>
    %A = sdir.alloc_stream() : !sdir.stream_array<2x6xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.stream_array<2x6xi32> -> !sdir.stream<2x6xi32>
        %a = sdir.get_access %A : !sdir.stream_array<2x6xi32> -> !sdir.stream<2x6xi32>
        // CHECK-NEXT: func @empty
        func @empty(%x: !sdir.stream<i32>) -> i1{
            %0 = constant 0 : i32
            %l = sdir.stream_length %x : !sdir.stream<i32> -> i32
            %isZero = cmpi "eq", %l, %0 : i32
            return %isZero : i1
        }
        // CHECK: sdir.consume
        sdir.consume{num_pes=5, condition=@empty} (%a : !sdir.stream<2x6xi32>) -> (pesid: %p, elem: %e) {
        }
    }
}
