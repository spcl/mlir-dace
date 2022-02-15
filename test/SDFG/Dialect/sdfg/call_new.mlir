sdfg.sdfg{entry=@state_0} @sdfg_0 {
  sdfg.state @state_0 {
    %N = sdfg.alloc() : !sdfg.array<i32>

    %res = sdfg.sdfg{entry=@state_1} @sdfg_1(%N as %a: !sdfg.array<i32>) -> !sdfg.array<i32> {
      // Optionally: Order decides return order
      %res = sdfg.alloc{retPos=0}() : !sdfg.array<i32>

      sdfg.state @state_1 {
        // Loads & Stores
      }
    }
  }
} 


sdfg.sdfg{entry=@state_0} @sdfg_0 {
  sdfg.state @state_0 {
    %N = sdfg.alloc() : !sdfg.array<i32>

    %res = sdfg.sdfg{entry=@state_1} @sdfg_1(%N as %a: !sdfg.array<i32>) -> !sdfg.array<i32> {
      sdfg.state @state_1 {
        // Loads & Stores
        sdir.return %arrayName
      }
    }
  }
}


sdfg.sdfg{entry=@state_0} @sdfg_0 {
  sdfg.state @state_0 {
    %N = sdfg.alloc() : !sdfg.array<i32>
    %res = sdfg.alloc() : !sdfg.array<i32>

    sdfg.sdfg{entry=@state_1} @sdfg_1(%N as %a: !sdfg.array<i32>) -> (%res as %b: !sdfg.array<i32>) {
      sdfg.state @state_1 {
        // Loads & Stores
      }
    }
  }
}

sdfg.sdfg{entry=@state_0} @sdfg_0 {
  sdfg.state @state_0 {
    %N = sdfg.alloc() : !sdfg.array<i32>
    %res = sdfg.alloc() : !sdfg.array<i32>

    sdfg.sdfg{entry=@state_1} @sdfg_1(%N as %a: !sdfg.array<i32>, returns %res as %b: !sdfg.array<i32>) {
      sdfg.state @state_1 {
        // Loads & Stores
      }
    }
  }
}
