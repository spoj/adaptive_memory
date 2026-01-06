fn main() {
    // For 69 edges, each neighbor has strength 1.0
    let num_edges = 69;

    // Each edge has raw strength 1.0
    // ln_1p(1.0) = 0.693
    let ln1p_val = 1.0_f64.ln_1p();
    println!("ln_1p(1.0) = {:.6}", ln1p_val);

    // Total across all neighbors
    let total = ln1p_val * num_edges as f64;
    println!("Total for {} neighbors = {:.6}", num_edges, total);

    // Each neighbor gets:
    let per_neighbor = ln1p_val / total;
    println!("Per neighbor (normalized) = {:.6}", per_neighbor);

    // Energy = seed_energy * energy_decay * per_neighbor
    let energy = 1.0 * 0.5 * per_neighbor;
    println!("Energy per neighbor = {:.6}", energy);
    println!("Threshold = 0.01, Above? {}", energy > 0.01);

    // But the actual energy was 0.0101. Why?
    // Let's solve: what normalized value gives 0.0101?
    let actual_energy = 0.010144927536231882;
    let actual_normalized = actual_energy / (1.0 * 0.5);
    println!("\nActual normalized = {:.6}", actual_normalized);
    println!("Expected normalized = {:.6}", per_neighbor);
    println!("Ratio = {:.4}", actual_normalized / per_neighbor);
}
