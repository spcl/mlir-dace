// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<12x45xi32>

  sdfg.state @state_0 {
    %a_1 = sdfg.load %A[6, 12] : !sdfg.array<12x45xi32> -> i32

    %res = sdfg.tasklet(%a_1: i32) -> (i32) {
        %z = arith.addi %a_1, %a_1 : i32
        sdfg.return %z : i32
    }

    sdfg.store %res, %A[6, 12] : i32 -> !sdfg.array<12x45xi32>
  }
} 
