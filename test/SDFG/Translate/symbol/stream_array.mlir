// RUN: sdfg-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    sdfg.state @state_0{
        sdfg.alloc_symbol("N")
        %a = sdfg.alloc() : !sdfg.stream_array<sym("N")xi32>
    }
}
