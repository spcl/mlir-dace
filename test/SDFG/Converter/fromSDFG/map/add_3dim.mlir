// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<2x6x8xi32>
  %B = sdfg.alloc() : !sdfg.array<2x6x8xi32>
  %C = sdfg.alloc() : !sdfg.array<2x6x8xi32>

  sdfg.state @state_0 {
    sdfg.map (%i, %j, %g) = (0, 0, 0) to (2, 2, 2) step (1, 1, 1) {
      %a_ijg = sdfg.load %A[%i, %j, %g] : !sdfg.array<2x6x8xi32> -> i32
      %b_ijg = sdfg.load %B[%i, %j, %g] : !sdfg.array<2x6x8xi32> -> i32

      %res = sdfg.tasklet(%a_ijg: i32, %b_ijg: i32) -> (i32) {
        %z = arith.addi %a_ijg, %b_ijg : i32
        sdfg.return %z : i32
      }

      sdfg.store %res, %C[%i, %j, %g] : i32 -> !sdfg.array<2x6x8xi32>
    }
  }
}
