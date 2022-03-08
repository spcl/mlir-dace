// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<8x16x4xi32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.array<8x16x4xi32> -> !sdfg.memlet<8x16x4xi32>
        %a_s = sdfg.subview %a[3, 4, 2][1, 6, 3][1, 1, 1] : !sdfg.memlet<8x16x4xi32> -> !sdfg.memlet<6x3xi32>
    }
}
