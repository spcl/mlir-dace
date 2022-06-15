// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<index>) {
  %A = sdfg.alloc() : !sdfg.array<2x6xi32>
  %B = sdfg.alloc() : !sdfg.array<2x6xi32>
  %C = sdfg.alloc() : !sdfg.array<2x6xi32>

  sdfg.state @state_0 {
    %c = sdfg.load %r[] : !sdfg.array<index> -> index

    sdfg.map (%i, %j) = (%c, %c) to (%c, %c) step (1, 1) {
      %a_ij = sdfg.load %A[0, 0] : !sdfg.array<2x6xi32> -> i32
      %b_ij = sdfg.load %B[0, 0] : !sdfg.array<2x6xi32> -> i32

      %res = sdfg.tasklet(%a_ij: i32, %b_ij: i32) -> (i32) {
        %z = arith.addi %a_ij, %b_ij : i32
        sdfg.return %z : i32
      }

      sdfg.store %res, %C[0, 0] : i32 -> !sdfg.array<2x6xi32>
    }
  }
}
