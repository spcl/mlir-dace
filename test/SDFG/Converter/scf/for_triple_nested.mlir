// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @kernel_2mm(%arg6: memref<?x900xf64>) {
  %4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg11 = %c0 to %4 step %c1 {
    scf.for %arg12 = %c0 to %4 step %c1 {
      scf.for %arg13 = %c0 to %4 step %c1 {
        %6 = memref.load %arg6[%arg11, %arg13] : memref<?x900xf64>
        memref.store %6, %arg6[%arg11, %arg12] : memref<?x900xf64>
      }
    }
  }
  return
}
