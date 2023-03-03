// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main() -> (index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %sum = scf.for %iv = %c0 to %c1 step %c1 iter_args(%sum_iter = %c0) -> (index) {
    %sum_next = arith.addi %sum_iter, %iv : index
    scf.yield %sum_next : index
  }

  return %sum : index
}
