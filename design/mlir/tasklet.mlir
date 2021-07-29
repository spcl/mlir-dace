sdir.tasklet{outputs=["c", "d"]} @add(
    %a: i32, %b: i32) -> (i32, i32) 
{
    %c = addi %a, %b : i32
    %d = subi %a, %b : i32
    sdir.return %c, %d : i32, i32
}
