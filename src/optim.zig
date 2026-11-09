const std = @import("std");
const kernels = @import("kernels.zig");

pub fn sgd(lr: f32, params: anytype) void {
    inline for (params) |p| {
        const E = @TypeOf(p.value).Elem;
        const Ctx = struct { w: []E, g: []const E, lr: E };
        const body = struct {
            fn run(c: Ctx, lo: usize, hi: usize) void {
                for (c.w[lo..hi], c.g[lo..hi]) |*w, g| w.* -= c.lr * g;
            }
        }.run;
        kernels.parallelFor(p.value.data.len, 1 << 14, Ctx{ .w = p.value.data, .g = p.grad.data, .lr = lr }, body);
    }
}

const testing = std.testing;
const Tensor = @import("tensor.zig").Tensor;
const Var = @import("autograd.zig").Var;

test "sgd moves weights against the gradient, also across parallel chunks" {
    const gpa = testing.allocator;
    const Small = Tensor(f32, &.{3});
    const Big = Tensor(f32, &.{100003});

    var small: Var(Small) = .{ .value = try Small.fromSlice(gpa, &.{ 1, 2, 3 }), .grad = try Small.fromSlice(gpa, &.{ 10, 20, 30 }), .needs_grad = true };
    defer small.value.deinit(gpa);
    defer small.grad.deinit(gpa);
    var big: Var(Big) = .{ .value = try Big.full(gpa, 1), .grad = try Big.empty(gpa), .needs_grad = true };
    defer big.value.deinit(gpa);
    defer big.grad.deinit(gpa);
    for (big.grad.data, 0..) |*g, i| g.* = @floatFromInt(i % 11);

    kernels.parallel = .{ .io = testing.io, .threads = 4 };
    defer kernels.parallel = .{};
    sgd(0.1, .{ small, big });

    try testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, small.value.data);
    for (big.value.data, 0..) |w, i| {
        try testing.expectApproxEqAbs(1.0 - 0.1 * @as(f32, @floatFromInt(i % 11)), w, 1e-6);
    }
}
