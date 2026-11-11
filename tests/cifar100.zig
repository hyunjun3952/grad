const std = @import("std");
const lib = @import("grad");
const load = lib.cifar100.load;
const cifar = lib.cifar;

const testing = std.testing;

test "records carry a coarse and a fine label, and the fine one is kept" {
    var entries: [2 * (2 + 3072)]u8 = undefined;
    @memset(&entries, 0);
    entries[0] = 19;
    entries[1] = 99;
    entries[2 + 3072] = 3;
    entries[2 + 3072 + 1] = 42;
    const data = try cifar.parseAs(testing.allocator, &.{&entries}, cifar.hundred);
    defer data.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, &.{ 99, 42 }, data.labels);
    entries[1] = 100;
    try testing.expectError(error.BadCifarFile, cifar.parseAs(testing.allocator, &.{&entries}, cifar.hundred));
    try testing.expectError(error.BadCifarFile, cifar.parseAs(testing.allocator, &.{entries[0..3000]}, cifar.hundred));
}

test "the real CIFAR-100 files, when present, have the documented shape" {
    const test_set = load(testing.allocator, testing.io, "data/cifar100", "t10k") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer test_set.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 10000), test_set.count());
    var counts: [100]usize = @splat(0);
    for (test_set.labels) |l| counts[l] += 1;
    for (counts) |n| try testing.expectEqual(@as(usize, 100), n);
}
