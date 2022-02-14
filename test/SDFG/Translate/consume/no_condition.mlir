// RUN: sdfg-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream_array<2x6xi32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.stream_array<2x6xi32> -> !sdfg.stream<2x6xi32>
        sdfg.consume{num_pes=5} (%a : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}