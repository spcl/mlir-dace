// XFAIL: *
// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<i32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
        %a_1 = sdir.load %a[] : !sdir.memlet<i32> -> i32
        sdir.store %a_1, %a[] : i32 -> !sdir.memlet<i32>
    }
}
