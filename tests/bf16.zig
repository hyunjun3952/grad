const std = @import("std");
const lib = @import("grad");
const decode = lib.math.bf16.decode;
const encode = lib.math.bf16.encode;

test "bf16 rounds to nearest even and keeps the f32 range" {
    try std.testing.expectEqual(@as(f32, 1), decode(encode(1)));
    try std.testing.expectEqual(@as(f32, -2.5), decode(encode(-2.5)));
    try std.testing.expectEqual(@as(u16, 0x3F80), encode(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), encode(1.00195312));
    try std.testing.expectEqual(@as(u16, 0x3F82), encode(1.01171875));
    try std.testing.expectApproxEqRel(@as(f32, 1e-30), decode(encode(1e-30)), 1e-2);
    try std.testing.expectApproxEqRel(@as(f32, 3e38), decode(encode(3e38)), 1e-2);
    try std.testing.expect(std.math.isInf(decode(encode(std.math.inf(f32)))));
}
