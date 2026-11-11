const std = @import("std");
const lib = @import("grad");
const apply = lib.math.dropout.apply;

test "the kept fraction follows the keep probability and kept values are scaled" {
    var src: [20000]f32 = undefined;
    var dst: [20000]f32 = undefined;
    @memset(&src, 1);
    for (0..src.len) |t| apply(t, &src, &dst, 7, 0.8, false);
    var kept: usize = 0;
    for (dst) |v| {
        if (v != 0) {
            kept += 1;
            try std.testing.expectApproxEqAbs(@as(f32, 1.25), v, 1e-6);
        }
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), @as(f32, @floatFromInt(kept)) / src.len, 0.01);
}
