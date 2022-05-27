// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> () {
  %A = sdfg.alloc() : !sdfg.stream<i32>
  %C = sdfg.alloc() : !sdfg.array<6xi32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=5} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {

      sdfg.map (%i) = (0) to (2) step (1) {
        %res = sdfg.tasklet(%e: i32, %i: index) -> (i32) {
          %0 = arith.index_cast %i : index to i32
          %res = arith.addi %e, %0 : i32
          sdfg.return %res : i32
        }

        sdfg.store{wcr="add"} %res, %C[%i] : i32 -> !sdfg.array<6xi32>
      }
    }
  }
}
