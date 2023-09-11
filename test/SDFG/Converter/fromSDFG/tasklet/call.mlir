// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<i32>

  sdfg.state @state_0{
    %1 = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }

    %c = sdfg.tasklet(%1: i32) -> (i32) {
      %c = arith.addi %1, %1 : i32
      sdfg.return %c : i32
    }

    sdfg.store %c, %A[] : i32 -> !sdfg.array<i32>
  }
}
