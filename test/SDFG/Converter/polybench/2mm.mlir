// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x900xf64>, %arg7: memref<?x1100xf64>, %arg8: memref<?x900xf64>, %arg9: memref<?x1200xf64>, %arg10: memref<?x1200xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = arith.index_cast %arg1 : i32 to index
  %1 = arith.index_cast %arg2 : i32 to index
  %2 = arith.index_cast %arg3 : i32 to index
  %3 = arith.index_cast %arg1 : i32 to index
  %4 = arith.index_cast %arg0 : i32 to index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg11 = %c0 to %4 step %c1 {
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    scf.for %arg12 = %c0_2 to %0 step %c1_3 {
      memref.store %cst, %arg6[%arg11, %arg12] : memref<?x900xf64>
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg13 = %c0_4 to %1 step %c1_5 {
        %6 = memref.load %arg7[%arg11, %arg13] : memref<?x1100xf64>
        %7 = arith.mulf %arg4, %6 : f64
        %8 = memref.load %arg8[%arg13, %arg12] : memref<?x900xf64>
        %9 = arith.mulf %7, %8 : f64
        %10 = memref.load %arg6[%arg11, %arg12] : memref<?x900xf64>
        %11 = arith.addf %10, %9 : f64
        memref.store %11, %arg6[%arg11, %arg12] : memref<?x900xf64>
      }
    }
  }
  %5 = arith.index_cast %arg0 : i32 to index
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  scf.for %arg11 = %c0_0 to %5 step %c1_1 {
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    scf.for %arg12 = %c0_2 to %2 step %c1_3 {
      %6 = memref.load %arg10[%arg11, %arg12] : memref<?x1200xf64>
      %7 = arith.mulf %6, %arg5 : f64
      memref.store %7, %arg10[%arg11, %arg12] : memref<?x1200xf64>
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg13 = %c0_4 to %3 step %c1_5 {
        %8 = memref.load %arg6[%arg11, %arg13] : memref<?x900xf64>
        %9 = memref.load %arg9[%arg13, %arg12] : memref<?x1200xf64>
        %10 = arith.mulf %8, %9 : f64
        %11 = memref.load %arg10[%arg11, %arg12] : memref<?x1200xf64>
        %12 = arith.addf %11, %10 : f64
        memref.store %12, %arg10[%arg11, %arg12] : memref<?x1200xf64>
      }
    }
  }
  return
}
