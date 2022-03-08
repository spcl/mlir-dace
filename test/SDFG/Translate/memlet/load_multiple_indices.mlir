// XFAIL: *
// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<12x45xi32>
    sdfg.state @state_0 {
        sdfg.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdfg.return %0 : index
        }
        %0 = sdfg.call @zero() : () -> index
        %a = sdfg.get_access %A : !sdfg.array<12x45xi32> -> !sdfg.memlet<12x45xi32>
        %a_1 = sdfg.load %a[%0, %0] : !sdfg.memlet<12x45xi32> -> i32
        sdfg.store %a_1, %a[%0, %0] : i32 -> !sdfg.memlet<12x45xi32>
    }
} 
