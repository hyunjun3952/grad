const std = @import("std");
const builtin = @import("builtin");

pub const Kind = enum(u32) { sigmoid, tanh, gelu };

inline fn exponential(x: f32) f32 {
    if (builtin.cpu.arch.isNvptx()) return asm ("ex2.approx.f32 %[out], %[in];"
        : [out] "=f" (-> f32),
        : [in] "f" (x * 1.442695041),
    );
    return @exp(x);
}

inline fn logistic(x: f32) f32 {
    return 1 / (1 + exponential(-x));
}

const gelu_scale = 0.7978845608;
const gelu_cubic = 0.044715;

inline fn geluInner(x: f32) f32 {
    const t = gelu_scale * (x + gelu_cubic * x * x * x);
    return 2 * logistic(2 * t) - 1;
}

pub inline fn forward(kind: Kind, x: f32) f32 {
    return switch (kind) {
        .sigmoid => logistic(x),
        .tanh => 2 * logistic(2 * x) - 1,
        .gelu => 0.5 * x * (1 + geluInner(x)),
    };
}

pub inline fn derivative(kind: Kind, x: f32, y: f32) f32 {
    return switch (kind) {
        .sigmoid => y * (1 - y),
        .tanh => 1 - y * y,
        .gelu => blk: {
            const inner = geluInner(x);
            const slope = gelu_scale * (1 + 3 * gelu_cubic * x * x);
            break :blk 0.5 * (1 + inner) + 0.5 * x * (1 - inner * inner) * slope;
        },
    };
}

pub fn apply(x: []const f32, y: []f32, kind: Kind, lo: usize, hi: usize) void {
    for (x[lo..hi], y[lo..hi]) |v, *out| out.* = forward(kind, v);
}

pub fn applyBackward(x: []const f32, y: []const f32, g_out: []const f32, g_in: []f32, kind: Kind, lo: usize, hi: usize) void {
    for (x[lo..hi], y[lo..hi], g_out[lo..hi], g_in[lo..hi]) |v, w, g, *gx| gx.* += g * derivative(kind, v, w);
}

test "activations match reference values and their derivatives match finite differences" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), forward(.sigmoid, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.880797), forward(.sigmoid, 2), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.761594), forward(.tanh, 1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.761594), forward(.tanh, -1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841192), forward(.gelu, 1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.158808), forward(.gelu, -1), 1e-5);
    inline for (.{ Kind.sigmoid, Kind.tanh, Kind.gelu }) |kind| {
        for ([_]f32{ -3, -1.2, -0.3, 0.4, 1.5, 3 }) |x| {
            const h: f32 = 1e-2;
            const numeric = (forward(kind, x + h) - forward(kind, x - h)) / (2 * h);
            try std.testing.expectApproxEqAbs(numeric, derivative(kind, x, forward(kind, x)), 2e-3);
        }
    }
}
