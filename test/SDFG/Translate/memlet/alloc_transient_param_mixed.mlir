// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0{
    %n = sdfg.tasklet() -> (index) {
      %5 = arith.constant 5 : index
      sdfg.return %5 : index
    }

    %m = sdfg.tasklet() -> (index) {
      %20 = arith.constant 20 : index
      sdfg.return %20 : index
    }

    %a = sdfg.alloc{transient}(%n, %m) : !sdfg.array<?x6x?xi32>
  }
}
