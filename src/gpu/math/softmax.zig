const activation = @import("activation.zig");
const bf16 = @import("bf16.zig");

pub inline fn limitOf(row: usize, cols: usize, period: usize) usize {
    return if (period == 0) cols else row % period + 1;
}

// Rows of x, y, dy and dx may be f32 or bf16 (E); the sums run in f32. The row is softmax(scale * x) with scale above zero,
// so the largest x also gives the largest scaled value.
// NOTE: a bf16 row recomputes the exponentials in the last pass instead of storing them, because rounding them before the division would add error.
pub inline fn forwardRow(comptime E: type, x: [*]const E, y: [*]E, cols: usize, limit: usize, scale: f32) void {
    var largest = bf16.load(E, x, 0);
    for (1..limit) |j| largest = @max(largest, bf16.load(E, x, j));
    var total: f32 = 0;
    for (0..limit) |j| {
        const e = activation.exponential(scale * (bf16.load(E, x, j) - largest));
        if (E == f32) y[j] = e;
        total += e;
    }
    const inverse = 1 / total;
    for (0..limit) |j| {
        if (E == f32) {
            y[j] *= inverse;
        } else {
            bf16.store(E, y, j, activation.exponential(scale * (bf16.load(E, x, j) - largest)) * inverse);
        }
    }
    for (limit..cols) |j| bf16.store(E, y, j, 0);
}

pub inline fn backwardRow(comptime E: type, y: [*]const E, dy: [*]const E, dx: [*]E, cols: usize, scale: f32) void {
    var dot: f32 = 0;
    for (0..cols) |j| dot += bf16.load(E, y, j) * bf16.load(E, dy, j);
    for (0..cols) |j| bf16.store(E, dx, j, bf16.load(E, dx, j) + scale * bf16.load(E, y, j) * (bf16.load(E, dy, j) - dot));
}
