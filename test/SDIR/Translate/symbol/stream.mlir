// RUN: sdir-translate --mlir-to-sdfg %s | not python %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node
sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.stream_array<12xi32>
    sdir.state @state_0 {
        sdir.alloc_symbol("N")
        %a = sdir.get_access %A : !sdir.stream_array<12xi32> -> !sdir.stream<sym("N")xi32>
    }
}
