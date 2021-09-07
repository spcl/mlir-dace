sdir.tasklet @add(%a: i32, %b: i32) -> i32{
    %c = addi %a, %b : i32
    sdir.return %c
}

sdir.state @state_0{ 
    %a = sdir.get_access %A 
    %b = sdir.get_access %B 
    %c = sdir.get_access %C 
    
    %a_val = sdir.load %a[0]
    %b_val = sdir.load %b[0]
    %c_val = sdir.call @add(%a, %b)

    sdir.store %c_val, %c[0]
}
