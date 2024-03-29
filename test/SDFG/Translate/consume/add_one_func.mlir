// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.stream<i32>
  %C = sdfg.alloc() : !sdfg.array<6xi32>

  sdfg.state @state_0 {
    func.func @empty(%x: !sdfg.stream<i32>) -> i1{
      %0 = arith.constant 0 : i32
      %length = sdfg.stream_length %x : !sdfg.stream<i32> -> i32
      %isZero = arith.cmpi "eq", %length, %0 : i32
      return %isZero : i1
    }

    sdfg.consume{num_pes=5, condition=@empty} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
      %res = sdfg.tasklet(%e: i32) -> (i32) {
        %1 = arith.constant 1 : i32
        %res = arith.addi %e, %1 : i32
        sdfg.return %res : i32
      }

      sdfg.store{wcr="add"} %res, %C[0] : i32 -> !sdfg.array<6xi32>
    }
  }
}
