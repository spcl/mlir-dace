// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.stream_array<12xi32>
    sdfg.state @state_0 {
        sdfg.alloc_symbol("N")
        %a = sdfg.get_access %A : !sdfg.stream_array<12xi32> -> !sdfg.stream<sym("N")xi32>
    }
}
