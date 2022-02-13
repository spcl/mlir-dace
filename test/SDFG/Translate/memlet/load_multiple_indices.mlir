// XFAIL: *
// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py

sdir.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdir.alloc() : !sdir.array<12x45xi32>
    sdir.state @state_0 {
        sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        %0 = sdir.call @zero() : () -> index
        %a = sdir.get_access %A : !sdir.array<12x45xi32> -> !sdir.memlet<12x45xi32>
        %a_1 = sdir.load %a[%0, %0] : !sdir.memlet<12x45xi32> -> i32
        sdir.store %a_1, %a[%0, %0] : i32 -> !sdir.memlet<12x45xi32>
    }
} 
