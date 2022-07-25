// RUN: sdfg-opt --convert-to-sdfg %s
memref.global @arr : memref<900xi32>

func.func private @kernel(%1: index){
  %22 = memref.get_global @arr : memref<900xi32>
  %8 = memref.load %22[%1] : memref<900xi32>
  %c2 = arith.addi %8, %8 : i32
  %23 = memref.get_global @arr : memref<900xi32>
  memref.store %c2, %23[%1] : memref<900xi32>
  return
}
