// RUN: sdir-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py %s 2>&1 | FileCheck %s
// CHECK: Isolated node
sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc_stream() : !sdir.stream_array<2x6xi32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.stream_array<2x6xi32> -> !sdir.stream<2x6xi32>
        builtin.func @empty(%x: !sdir.stream<2x6xi32>) -> i1{
            %0 = arith.constant 0 : i32
            %length = sdir.stream_length %x : !sdir.stream<2x6xi32> -> i32
            %isZero = arith.cmpi "eq", %length, %0 : i32
            return %isZero : i1
        }
        sdir.consume{condition=@empty} (%a : !sdir.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}
