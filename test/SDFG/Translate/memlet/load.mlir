// RUN: sdfg-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<i32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.array<i32> -> !sdfg.memlet<i32>
        %a_1 = sdfg.load %a[] : !sdfg.memlet<i32> -> i32
        sdfg.store %a_1, %a[] : i32 -> !sdfg.memlet<i32>
    }
}
