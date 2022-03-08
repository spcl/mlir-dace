// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<12x12xi32>
    sdfg.state @state_0 {
        sdfg.alloc_symbol("N")
        sdfg.tasklet @one() -> i32{
            %1 = arith.constant 1 : i32
            sdfg.return %1 : i32
        }
        %1 = sdfg.call @one() : () -> i32
        %a = sdfg.get_access %A : !sdfg.array<12x12xi32> -> !sdfg.memlet<12x12xi32>
        sdfg.store %1, %a[0, sym("N")] : i32 -> !sdfg.memlet<12x12xi32>
    }
}
