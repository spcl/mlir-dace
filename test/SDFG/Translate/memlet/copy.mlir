// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<i32>
    %B = sdfg.alloc() : !sdfg.array<i32>
    sdfg.state @state_0 {
        sdfg.copy %A -> %B : !sdfg.array<i32>
    }
}
