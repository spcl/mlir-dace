// RUN: sdfg-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %a = sdfg.alloc() : !sdfg.array<i32>

    sdfg.state @state_0{
    }
}
