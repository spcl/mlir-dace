// RUN: not sdir-opt %s 2>&1 | FileCheck %s
// CHECK: expected ')'

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.stream<2x6xi32>

    sdir.state @state_0 {

        builtin.func @empty(%x: !sdir.stream<2x6xi32>) -> i1{
            %0 = arith.constant 0 : i32
            %l = sdir.stream_length %x : !sdir.stream<2x6xi32> -> i32
            %isZero = arith.cmpi "eq", %l, %0 : i32
            return %isZero : i1
        }

        sdir.consume{num_pes=5, condition=@empty} (%A : !sdir.stream<2x6xi32>) -> (pe: %p, elem: %e, %b) {
        }
    }
}
