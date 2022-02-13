// RUN: sdfg-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<i32>
    %B = sdfg.alloc() : !sdfg.array<i32>
    sdfg.state @state_0 {
        %a = sdfg.get_access %A : !sdfg.array<i32> -> !sdfg.memlet<i32>
        %b = sdfg.get_access %B : !sdfg.array<i32> -> !sdfg.memlet<i32>
        sdfg.copy %a -> %b : !sdfg.memlet<i32>
    }
}
