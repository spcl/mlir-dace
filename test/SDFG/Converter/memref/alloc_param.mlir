// RUN: sdfg-opt --convert-to-sdfg %s
func.func private @kernel_2mm(%1: index){
  %c0 = arith.constant 0 : index
  %arg6 = memref.alloc(%c0) : memref<?x900xi32>
  %8 = memref.load %arg6[%c0, %1] : memref<?x900xi32>
  %c2 = arith.addi %8, %8 : i32
  memref.store %c2, %arg6[%1, %1] : memref<?x900xi32>
  return
}
