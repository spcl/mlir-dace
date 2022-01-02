// XFAIL: *
// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<12xi32>
    sdir.state @state_0 {
        sdir.alloc_symbol("N")
        %a = sdir.get_access %A : !sdir.array<12xi32> -> !sdir.memlet<sym("N")xi32>
    }
}
