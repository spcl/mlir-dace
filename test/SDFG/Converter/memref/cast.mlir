// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt

func.func private @main(%1: index){
  %c0 = arith.constant 0 : index
  %A = memref.alloc() : memref<5x900xi32>
  %B = memref.cast %A : memref<5x900xi32> to memref<?x?xi32>
  %B_val = memref.load %B[%c0, %1] : memref<?x?xi32>
  %B_val_new = arith.addi %B_val, %B_val : i32
  memref.store %B_val_new, %B[%1, %1] : memref<?x?xi32>
  return
}
