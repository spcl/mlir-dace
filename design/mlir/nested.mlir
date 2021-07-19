sdir.sdfg{entry=@state_0, outputs=["C"]} @name(%A: !sdir.memlet<i32>, 
                                               %C: !sdir.memlet<i32>) 
{ 
    sdir.edge{assign=["i = 1"]} @state_0 -> @state_1
    sdir.edge{condition="i > 1"} @state_1 -> @state_2
    sdir.edge{condition="i <= 1"} @state_1 -> @state_3
}