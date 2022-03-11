// XFAIL: *
// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg{entry=@state_0} {
    %A = sdfg.alloc() : !sdfg.array<12x45xi32>

    sdfg.state @state_0 {

        %0 = sdfg.tasklet() -> (index) {
                %0 = arith.constant 0 : index
                sdfg.return %0 : index
            }

        %a_1 = sdfg.load %A[%0, %0] : !sdfg.array<12x45xi32> -> i32
        sdfg.store %a_1, %A[%0, %0] : i32 -> !sdfg.array<12x45xi32>
    }
} 
