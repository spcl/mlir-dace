// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<5x3xi32>

    sdfg.state @state_0 {
        sdfg.alloc_symbol("N")
        %a_1 = sdfg.load %A[sym("N"), 0] : !sdfg.array<5x3xi32> -> i32
    }
}
