// XFAIL: *
// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> () {
  %A = sdfg.alloc() : !sdfg.stream<i32>
  %C = sdfg.alloc() : !sdfg.array<i32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=5} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
      %res = sdfg.tasklet(%e: i32, %p: index) -> (i32) {
        %0 = arith.index_cast %p : index to i32
        %res = arith.addi %e, %0 : i32
        sdfg.return %res : i32
      }

      sdfg.store{wcr="add"} %res, %C[] : i32 -> !sdfg.array<i32>
    }
  }
}
