// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%arg0: !sdfg.array<i32>, %arg1: !sdfg.array<i32>) {
  sdfg.state @state_0{
    %n:2 = sdfg.tasklet() -> (i32, i32) {
      %1 = arith.constant 1 : i32
      %2 = arith.addi %1, %1 : i32
      sdfg.return %1, %2 : i32, i32
    }

    sdfg.store %n#0, %arg0[] : i32 -> !sdfg.array<i32>
    sdfg.store %n#1, %arg1[] : i32 -> !sdfg.array<i32>
  }
}
