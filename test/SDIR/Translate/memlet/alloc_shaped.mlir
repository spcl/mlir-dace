// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %a = sdir.alloc() : !sdir.array<23x45x123xi32>

    sdir.state @state_0{
    }
} 
