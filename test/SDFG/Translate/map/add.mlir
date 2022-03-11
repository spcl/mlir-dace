// RUN: sdfg-translate --mlir-to-sdfg %s | not python3 %S/../import_translation_test.py 2>&1 | FileCheck %s
// CHECK: Isolated node

sdfg.sdfg{entry=@state_0} {
    %A = sdfg.alloc() : !sdfg.array<2x6xi32>
    %B = sdfg.alloc() : !sdfg.array<2x6xi32>
    %C = sdfg.alloc() : !sdfg.array<2x6xi32>

    sdfg.state @state_0 {
        sdfg.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
            %a_ij = sdfg.load %A[%i, %j] : !sdfg.array<2x6xi32> -> i32
            %b_ij = sdfg.load %B[%i, %j] : !sdfg.array<2x6xi32> -> i32

            %res = sdfg.tasklet(%a_ij: i32, %b_ij: i32) -> (i32) {
                %z = arith.addi %a_ij, %b_ij : i32
                sdfg.return %z : i32
            }

            sdfg.store %res, %C[%i, %j] : i32 -> !sdfg.array<2x6xi32>
        }
    }
}
