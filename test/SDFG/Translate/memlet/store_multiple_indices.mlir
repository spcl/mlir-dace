// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Dangling out-connector

sdfg.sdfg{entry=@state_0} @sdfg_0 {
    %A = sdfg.alloc() : !sdfg.array<56x45xi32>

    sdfg.state @state_0 {
        %0 = sdfg.tasklet @zero() -> index{
                %0 = arith.constant 0 : index
                sdfg.return %0 : index
            }

        %1 = sdfg.tasklet @one() -> i32{
                %1 = arith.constant 1 : i32
                sdfg.return %1 : i32
            }

        sdfg.store %1, %A[%0, %0] : i32 -> !sdfg.array<56x45xi32>
    }
} 
