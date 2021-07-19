sdir.tasklet @add(%a: i32, %b: i32) -> i32{
    %c = sdir.addi %a, %b : i32
    sdir.return %c
}

sdir.state @state_0{ 
    %a = sdir.get_access %A 
    %b = sdir.get_access %B 
    %c = sdir.get_access %C 
     
    %c = sdir.call @add(%a, %b)
}