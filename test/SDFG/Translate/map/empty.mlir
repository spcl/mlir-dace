// RUN: sdfg-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<2x6xi32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.array<2x6xi32> -> !sdfg.memlet<2x6xi32>
        sdfg.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
        }
    }
}
