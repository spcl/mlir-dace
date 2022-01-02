// XFAIL: *
// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<8x16x4xi32>
    sdir.state @state_0 {
        %a = sdir.get_access %A : !sdir.array<8x16x4xi32> -> !sdir.memlet<8x16x4xi32>
        %a_s = sdir.subview %a[3, 4, 2][1, 6, 3][1, 1, 1] : !sdir.memlet<8x16x4xi32> -> !sdir.memlet<6x3xi32>
    }
}
