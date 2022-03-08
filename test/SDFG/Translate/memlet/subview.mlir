// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<8x16x4xi32>

    sdfg.state @state_0 {
        %a_s = sdfg.subview %A[3, 4, 2][1, 6, 3][1, 1, 1] : !sdfg.array<8x16x4xi32> -> !sdfg.array<6x3xi32>
    }
}
