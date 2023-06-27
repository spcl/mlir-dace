// RUN: sdfg-opt --lower-sdfg %s
module {
  sdfg.sdfg {entry = @init_0} () -> (%arg0: !sdfg.array<i32>){
    %0 = sdfg.alloc {name = "_load_tmp_5", transient} () : !sdfg.array<i32>
    %1 = sdfg.alloc {name = "_alloc_tmp_3", transient} () : !sdfg.array<1xi32>
    %2 = sdfg.alloc {name = "_constant_tmp_2", transient} () : !sdfg.array<index>
    sdfg.state @init_0{
    }
    sdfg.state @constant_1{
      %3 = sdfg.tasklet () -> (index){
        %c0 = arith.constant 0 : index
        sdfg.return %c0 : index
      }
      sdfg.store %3, %2[] : index -> !sdfg.array<index>
      %4 = sdfg.load %2[] : !sdfg.array<index> -> index
    }
    sdfg.state @alloc_init_4{
    }
    sdfg.state @load_6{
      %3 = sdfg.load %2[] : !sdfg.array<index> -> index
      %4 = sdfg.load %1[%3] : !sdfg.array<1xi32> -> i32
      sdfg.store %4, %0[] : i32 -> !sdfg.array<i32>
      %5 = sdfg.load %0[] : !sdfg.array<i32> -> i32
    }
    sdfg.state @return_7{
      %3 = sdfg.load %0[] : !sdfg.array<i32> -> i32
      sdfg.store %3, %arg0[] : i32 -> !sdfg.array<i32>
    }
    sdfg.edge {assign = [], condition = "1"} @init_0 -> @constant_1
    sdfg.edge {assign = [], condition = "1"} @constant_1 -> @alloc_init_4
    sdfg.edge {assign = [], condition = "1"} @alloc_init_4 -> @load_6
    sdfg.edge {assign = [], condition = "1"} @load_6 -> @return_7
  }
}

