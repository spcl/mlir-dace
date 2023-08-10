// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> () {
  %C = sdfg.alloc() : !sdfg.array<2xi32>

  sdfg.state @state_0 {
    sdfg.map (%i) = (0) to (2) step (1) {
      %a = sdfg.load %C[%i] : !sdfg.array<2xi32> -> i32

      %res = sdfg.tasklet(%a: i32) -> (index) {
        %0 = arith.index_cast %a : i32 to index
        sdfg.return %0 : index
      }

      sdfg.store{wcr="add"} %a, %C[%res] : i32 -> !sdfg.array<2xi32>
    }
  }
}
