// XFAIL: *
// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<2x12xi32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.array<2x12xi32> -> !sdir.memlet<2x12xi32>
        %b = sdir.view_cast %a : !sdir.memlet<2x12xi32> -> !sdir.memlet<2x12xi32>
    }
} 
