// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: func with invalid signature

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream<2x6xi32>

    sdfg.state @state_0 {
        builtin.func @empty(%x: !sdfg.stream<i32>) -> i1{
            %0 = arith.constant 0 : i32
            %l = sdfg.stream_length %x : !sdfg.stream<i32> -> i32
            %isZero = arith.cmpi "eq", %l, %0 : i32
            return %isZero : i1
        }

        sdfg.consume{num_pes=5, condition=@empty} (%A : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}
