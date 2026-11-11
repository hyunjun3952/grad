const std = @import("std");
const lib = @import("grad");
const deviation = lib.cifar.deviation;
const mean = lib.cifar.mean;
const pixels = lib.cifar.pixels;
const parse = lib.cifar.parse;
const record = lib.cifar.record;
const load = lib.cifar.load;

const testing = std.testing;

test "parse converts planar records to interleaved normalized pixels and keeps labels" {
    var entry: [2 * record]u8 = undefined;
    @memset(&entry, 0);
    entry[0] = 3;
    entry[1] = 255;
    entry[1 + 1024] = 102;
    entry[1 + 2048] = 51;
    entry[record] = 9;
    const data = try parse(testing.allocator, &.{&entry});
    defer data.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), data.count());
    try testing.expectEqual(@as(usize, pixels), data.pixels);
    try testing.expectEqualSlices(u8, &.{ 3, 9 }, data.labels);
    try testing.expectApproxEqAbs((1.0 - mean[0]) / deviation[0], data.images[0], 1e-5);
    try testing.expectApproxEqAbs((0.4 - mean[1]) / deviation[1], data.images[1], 1e-5);
    try testing.expectApproxEqAbs((0.2 - mean[2]) / deviation[2], data.images[2], 1e-5);
    try testing.expectApproxEqAbs((0 - mean[0]) / deviation[0], data.images[3], 1e-5);
}

test "parse rejects truncated files and labels out of range" {
    var entry: [record]u8 = undefined;
    @memset(&entry, 0);
    try testing.expectError(error.BadCifarFile, parse(testing.allocator, &.{entry[0 .. record - 1]}));
    entry[0] = 10;
    try testing.expectError(error.BadCifarFile, parse(testing.allocator, &.{&entry}));
}

test "the real CIFAR-10 files, when present, have the documented shape" {
    const train = load(testing.allocator, testing.io, "data/cifar", "train") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer train.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 50000), train.count());
    for (train.labels) |l| try testing.expect(l < 10);
    var counts: [10]usize = @splat(0);
    for (train.labels) |l| counts[l] += 1;
    for (counts) |n| try testing.expectEqual(@as(usize, 5000), n);
}
