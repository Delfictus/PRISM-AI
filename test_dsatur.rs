use ndarray::Array2;

fn main() {
    // Test DSATUR on a simple triangle (needs 3 colors)
    let mut adj = Array2::from_elem((3, 3), false);
    adj[[0, 1]] = true; adj[[1, 0]] = true;
    adj[[1, 2]] = true; adj[[2, 1]] = true;
    adj[[2, 0]] = true; adj[[0, 2]] = true;
    
    println!("Triangle graph adjacency:");
    println!("{:?}", adj);
    println!("This graph needs exactly 3 colors (it's a clique)");
}
