module {
  func private @kernel_2mm(%arg0: i32, %arg4: f64, %arg10: memref<?x1200xf64>) {
    %A = memref.alloc() : memref<100xi64>
    %0 = arith.constant 0 : index
    %1 = arith.constant 1 : index
    %99 = arith.constant 99 : index
    %2 = arith.constant 2 : i64
    
    scf.for %i = %0 to %99 step %1 {
      %22 = arith.addi %2, %2 : i64
      memref.store %22, %A[%i] : memref<100xi64>
    }
    
    return
  }
}
