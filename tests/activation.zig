const std = @import("std");
const lib = @import("grad");
const derivative = lib.math.activation.derivative;
const forward = lib.math.activation.forward;
const Kind = lib.math.activation.Kind;

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

test "pointwise functions match reference values" {
    const pointwise = lib.math.activation.pointwise;
    try std.testing.expectEqual(@as(f32, 3), pointwise(.sqrt, 9));
    try std.testing.expectEqual(@as(f32, 0.25), pointwise(.reciprocal, 4));
}
