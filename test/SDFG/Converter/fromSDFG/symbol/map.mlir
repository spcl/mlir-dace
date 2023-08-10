// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    sdfg.alloc_symbol("N")
      %A = sdfg.alloc() : !sdfg.array<2x6xi32>
      %B = sdfg.alloc() : !sdfg.array<2x6xi32>
      %C = sdfg.alloc() : !sdfg.array<2x6xi32>


    sdfg.map (%i, %j) = (0, 0) to (2, sym("N")) step (0, sym("N+2")) {
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
