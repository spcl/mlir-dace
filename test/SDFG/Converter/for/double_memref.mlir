// RUN: sdir-opt --convert-to-sdir %s
func private @kernel_2mm(%arg0: i32, %arg6: memref<?x900xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg11 = %c0 to %c1 step %c1 {
    %v = memref.load %arg6[%arg11, %arg11] : memref<?x900xi32>
    %v2 = arith.addi %v, %v : i32
    memref.store %v2, %arg6[%arg11, %arg11] : memref<?x900xi32>
  }
  return
}
