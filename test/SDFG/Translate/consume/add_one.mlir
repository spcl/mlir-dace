// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> () {
  %A = sdfg.alloc() : !sdfg.stream<i32>
  %C = sdfg.alloc() : !sdfg.array<6xi32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=5} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
      %res = sdfg.tasklet(%e: i32) -> (i32) {
        %1 = arith.constant 1 : i32
        %res = arith.addi %e, %1 : i32
        sdfg.return %res : i32
      }

      sdfg.store{wcr="add"} %res, %C[0] : i32 -> !sdfg.array<6xi32>
    }
  }
}
