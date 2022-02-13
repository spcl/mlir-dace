// RUN: sdfg-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0 {
        sdfg.sdfg{entry=@state_1} @sdfg_1(%a: !sdfg.memlet<i32>) -> i32 {
            sdfg.state @state_1 {
            }
        }

        %N = sdfg.alloc() : !sdfg.array<i32>
        %n = sdfg.get_access %N : !sdfg.array<i32> -> !sdfg.memlet<i32>
        %c = sdfg.call @sdfg_1(%n) : (!sdfg.memlet<i32>) -> i32

        %M = sdfg.alloc() : !sdfg.array<i32>
        %m = sdfg.get_access %M : !sdfg.array<i32> -> !sdfg.memlet<i32>
        sdfg.store %c, %m[] : i32 -> !sdfg.memlet<i32>
    }
} 
