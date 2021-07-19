sdir.sdfg{entry=@state_0} { 
    sdir.edge{assign=["i = 1"]} 
        @state_0 -> @state_1
    sdir.edge{condition="i > 1"} 
        @state_1 -> @state_2
    sdir.edge{condition="i <= 1"} 
        @state_1 -> @state_3
}