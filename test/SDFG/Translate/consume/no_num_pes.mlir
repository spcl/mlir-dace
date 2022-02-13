// RUN: sdfg-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream_array<2x6xi32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.stream_array<2x6xi32> -> !sdfg.stream<2x6xi32>
        builtin.func @empty(%x: !sdfg.stream<2x6xi32>) -> i1{
            %0 = arith.constant 0 : i32
            %length = sdfg.stream_length %x : !sdfg.stream<2x6xi32> -> i32
            %isZero = arith.cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }
        sdfg.consume{condition=@empty} (%a : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}
