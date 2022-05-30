// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> () {
  %A = sdfg.alloc() : !sdfg.stream<i32>
  %C = sdfg.alloc() : !sdfg.array<6xi32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=5} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
      %res = sdfg.tasklet(%p: index) -> (i32) {
        %0 = arith.index_cast %p : index to i32
        sdfg.return %0 : i32
      }

      %0 = sdfg.tasklet() -> (index) {
        %0 = arith.constant 0 : index
        sdfg.return %0 : index
      }

      sdfg.store{wcr="add"} %res, %C[%0] : i32 -> !sdfg.array<6xi32>
    }
  }
}
