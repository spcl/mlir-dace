// XFAIL: *
// RUN: sdfg-opt %s 

sdfg.sdfg{entry=@state_0} {
  sdfg.state @state_0{
    %a = sdfg.tasklet() -> i32 {
      %c = arith.constant 0 : i32
      sdfg.return %c : i32
    }

    %res = sdfg.tasklet(%a: i32, %a as %b: i32) -> i32 {
      %c = arith.addi %a, %a : i32
      sdfg.return %c : i32
    }
  }
}
