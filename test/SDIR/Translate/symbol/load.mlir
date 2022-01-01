// XFAIL: *
// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<5x3xi32>
    sdir.state @state_0 {
        sdir.alloc_symbol("N")
        %a = sdir.get_access %A : !sdir.array<5x3xi32> -> !sdir.memlet<5x3xi32>
        %a_1 = sdir.load %a[sym("N"), 0] : !sdir.memlet<5x3xi32> -> i32
    }
}
