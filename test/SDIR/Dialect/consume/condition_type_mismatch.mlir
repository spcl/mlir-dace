// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.stream<2x6xi32>
    %A = sdir.alloc() : !sdir.stream<2x6xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: builtin.func @empty
        builtin.func @empty(%x: !sdir.stream<i32>) -> i1{
            %0 = arith.constant 0 : i32
            %l = sdir.stream_length %x : !sdir.stream<i32> -> i32
            %isZero = arith.cmpi "eq", %l, %0 : i32
            return %isZero : i1
        }
        // CHECK: sdir.consume
        sdir.consume{num_pes=5, condition=@empty} (%A : !sdir.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}
