// RUN: sdir-opt --convert-to-sdir %s
func private @kernel_2mm(%arg4: f64, %arg6: memref<?x900xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg12 = %c0 to %c0 step %c1 {
    memref.store %arg4, %arg6[%arg12, %arg12] : memref<?x900xf64>
    scf.for %arg13 = %c0 to %c1 step %c1 {
     %6 = memref.load %arg6[%arg12, %arg13] : memref<?x900xf64>
      %7 = arith.mulf %arg4, %6 : f64
      memref.store %7, %arg6[%arg12, %arg12] : memref<?x900xf64>
    }
  }
  return
}

