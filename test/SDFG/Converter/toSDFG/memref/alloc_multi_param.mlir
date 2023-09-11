// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main(){
  %c0 = arith.constant 0 : index
  %arg6 = memref.alloc(%c0, %c0, %c0) : memref<?x?x?xi32>
  %8 = memref.load %arg6[%c0, %c0, %c0] : memref<?x?x?xi32>
  %c2 = arith.addi %8, %8 : i32
  memref.store %c2, %arg6[%c0, %c0, %c0] : memref<?x?x?xi32>
  return
}
