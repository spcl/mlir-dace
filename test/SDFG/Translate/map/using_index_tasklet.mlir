// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> () {
  %C = sdfg.alloc() : !sdfg.array<i32>
  %B = sdfg.alloc() : !sdfg.array<i32>

  sdfg.state @state_0 {
    sdfg.map (%i) = (0) to (2) step (1) {
      %a = sdfg.load %C[] : !sdfg.array<i32> -> i32

      %res = sdfg.tasklet(%i: index, %a: i32) -> (i32) {
        %0 = arith.index_cast %i : index to i32
        %res = arith.addi %a, %0 : i32
        sdfg.return %res : i32
      }

      sdfg.store{wcr="add"} %res, %C[] : i32 -> !sdfg.array<i32>
    }
  }
}
