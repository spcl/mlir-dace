// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> () {
  %A = sdfg.alloc() : !sdfg.stream<i32>
  %C = sdfg.alloc() : !sdfg.array<i32>
  %B = sdfg.alloc() : !sdfg.array<i32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=5} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
      sdfg.map (%i) = (0) to (2) step (1) {
        %res = sdfg.tasklet(%e: i32) -> (i32) {
          sdfg.return %e : i32
        }
        sdfg.store{wcr="add"} %res, %C[] : i32 -> !sdfg.array<i32>
      }
    }
  }
}
