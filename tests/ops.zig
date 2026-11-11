const std = @import("std");
const lib = @import("grad");
const sumTo = lib.ops.sumTo;
const add = lib.ops.add;
const log = lib.ops.log;
const div = lib.ops.div;
const rowsView = lib.ops.rowsView;
const softmax = lib.ops.softmax;
const neg = lib.ops.neg;
const Tensor = lib.tensor.Tensor;
const mul = lib.ops.mul;
const logSoftmax = lib.ops.logSoftmax;
const reduce = lib.ops.reduce;
const relu = lib.ops.relu;
const exp = lib.ops.exp;
const sub = lib.ops.sub;

const testing = std.testing;
const expectEqualSlices = testing.expectEqualSlices;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

fn testArena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(testing.allocator);
}

test "add same shape, result type is the plain shape type" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const T23 = Tensor(f32, &.{ 2, 3 });
    const a = try T23.fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    const c: T23 = try add(gpa, a, try T23.full(gpa, 10));
    try expectEqualSlices(f32, &.{ 11, 12, 13, 14, 15, 16 }, c.data);
}

test "broadcast row vector, column against row, inner dim, scalars, 3d" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });

    const row = try Tensor(f32, &.{3}).fromSlice(gpa, &.{ 10, 20, 30 });
    const c = try add(gpa, a, row);
    try expectEqual(@as(usize, 2), @TypeOf(c).shape[0]);
    try expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, c.data);

    const col = try Tensor(f32, &.{ 2, 1 }).fromSlice(gpa, &.{ 10, 20 });
    try expectEqualSlices(f32, &.{ 11, 12, 13, 24, 25, 26 }, (try add(gpa, a, col)).data);

    const outer = try add(gpa, col, try Tensor(f32, &.{ 1, 3 }).fromSlice(gpa, &.{ 1, 2, 3 }));
    try expectEqualSlices(f32, &.{ 11, 12, 13, 21, 22, 23 }, outer.data);

    const s0 = try Tensor(f32, &.{}).fromSlice(gpa, &.{2});
    try expectEqualSlices(f32, &.{ 2, 4, 6, 8, 10, 12 }, (try mul(gpa, a, s0)).data);
    try expectEqualSlices(f32, &.{ 2, 4, 6, 8, 10, 12 }, (try mul(gpa, s0, a)).data);

    const s11 = try Tensor(f32, &.{ 1, 1 }).fromSlice(gpa, &.{3});
    try expectEqualSlices(f32, &.{ 3, 6, 9, 12, 15, 18 }, (try mul(gpa, a, s11)).data);

    const x = try Tensor(f32, &.{ 2, 1, 2 }).fromSlice(gpa, &.{ 1, 2, 3, 4 });
    const y = try Tensor(f32, &.{ 3, 1 }).fromSlice(gpa, &.{ 10, 20, 30 });
    const z = try add(gpa, x, y);
    try expectEqual(@as(usize, 3), @TypeOf(z).rank);
    try expectEqualSlices(f32, &.{ 11, 12, 21, 22, 31, 32, 13, 14, 23, 24, 33, 34 }, z.data);
}

test "sub div neg relu exp log" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const a = try Tensor(f32, &.{4}).fromSlice(gpa, &.{ 1, 2, 4, 8 });
    const b = try Tensor(f32, &.{4}).fromSlice(gpa, &.{ 1, 1, 2, 2 });
    try expectEqualSlices(f32, &.{ 0, 1, 2, 6 }, (try sub(gpa, a, b)).data);
    try expectEqualSlices(f32, &.{ 1, 2, 2, 4 }, (try div(gpa, a, b)).data);
    try expectEqualSlices(f32, &.{ -1, -2, -4, -8 }, (try neg(gpa, a)).data);

    const r = try Tensor(f32, &.{4}).fromSlice(gpa, &.{ -2, -0.5, 0, 3 });
    try expectEqualSlices(f32, &.{ 0, 0, 0, 3 }, (try relu(gpa, r)).data);

    const e = try exp(gpa, try Tensor(f32, &.{2}).fromSlice(gpa, &.{ 0, 1 }));
    try expectApproxEqAbs(@as(f32, 1), e.data[0], 1e-6);
    try expectApproxEqAbs(@as(f32, std.math.e), e.data[1], 1e-5);

    const l = try log(gpa, try Tensor(f32, &.{2}).fromSlice(gpa, &.{ 1, std.math.e }));
    try expectApproxEqAbs(@as(f32, 0), l.data[0], 1e-6);
    try expectApproxEqAbs(@as(f32, 1), l.data[1], 1e-6);
}

test "reduce over everything, f32 sums are accumulated in f64" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    const total = try reduce(gpa, a, null, .sum);
    try expectEqual(@as(usize, 0), @TypeOf(total).rank);
    try expectEqual(@as(f32, 21), total.data[0]);
    try expectEqual(@as(f32, 3.5), (try reduce(gpa, a, null, .mean)).data[0]);

    const big = try Tensor(f32, &.{1 << 20}).full(gpa, 0.1);
    try expectApproxEqAbs(@as(f32, 104857.6), (try reduce(gpa, big, null, .sum)).data[0], 0.05);
    try expectApproxEqAbs(@as(f32, 104857.6), (try reduce(gpa, big, 0, .sum)).data[0], 0.05);
    try expectApproxEqAbs(@as(f32, 0.1), (try reduce(gpa, big, null, .mean)).data[0], 1e-6);
}

test "axis sums with trailing dimensions are also accumulated in f64" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const tall = try Tensor(f32, &.{ 1 << 20, 2 }).full(gpa, 0.1);
    const s = try reduce(gpa, tall, 0, .sum);
    try expectApproxEqAbs(@as(f32, 104857.6), s.data[0], 0.05);
    try expectApproxEqAbs(@as(f32, 104857.6), s.data[1], 0.05);
    try expectApproxEqAbs(@as(f32, 0.1), (try reduce(gpa, tall, 0, .mean)).data[0], 1e-6);

    const t = try sumTo(gpa, tall, &.{2});
    try expectApproxEqAbs(@as(f32, 104857.6), t.data[0], 0.05);
    try expectApproxEqAbs(@as(f32, 104857.6), t.data[1], 0.05);
}

test "reduce along an axis" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 5, 3, 9, 2, 4 });
    const s0 = try reduce(gpa, a, 0, .sum);
    try expectEqual(@as(usize, 1), @TypeOf(s0).rank);
    try expectEqualSlices(f32, &.{ 10, 7, 7 }, s0.data);
    try expectEqualSlices(f32, &.{ 9, 15 }, (try reduce(gpa, a, 1, .sum)).data);
    try expectEqualSlices(f32, &.{ 5, 3.5, 3.5 }, (try reduce(gpa, a, 0, .mean)).data);
    try expectEqualSlices(usize, &.{ 1, 0, 1 }, (try reduce(gpa, a, 0, .argmax)).data);
    try expectEqualSlices(usize, &.{ 1, 0 }, (try reduce(gpa, a, 1, .argmax)).data);

    const b = try Tensor(f32, &.{ 2, 2, 2 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    try expectEqualSlices(f32, &.{ 6, 8, 10, 12 }, (try reduce(gpa, b, 0, .sum)).data);
    try expectEqualSlices(f32, &.{ 4, 6, 12, 14 }, (try reduce(gpa, b, 1, .sum)).data);
    try expectEqualSlices(f32, &.{ 3, 7, 11, 15 }, (try reduce(gpa, b, 2, .sum)).data);
}

test "softmax and logSoftmax are stable for large inputs" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 1000, 1001, 1002 });
    const s = try softmax(gpa, a);
    try expectApproxEqAbs(@as(f32, 0.09003057), s.data[0], 1e-6);
    try expectApproxEqAbs(@as(f32, 0.24472847), s.data[1], 1e-6);
    try expectApproxEqAbs(@as(f32, 0.66524096), s.data[2], 1e-6);
    for (0..3) |i| try expectApproxEqAbs(s.data[i], s.data[3 + i], 1e-6);

    const ls = try logSoftmax(gpa, a);
    try expectApproxEqAbs(@as(f32, -2.4076059), ls.data[0], 1e-5);
    try expectApproxEqAbs(@as(f32, -0.4076059), ls.data[2], 1e-5);
    for (0..3) |i| try expectApproxEqAbs(ls.data[i], ls.data[3 + i], 1e-4);
}

test "rowsView aliases the original data" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const a = try Tensor(f32, &.{ 4, 3 }).fromSlice(gpa, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });
    const rows = rowsView(a, 1, 2);
    try expectEqual(@as(usize, 2), @TypeOf(rows).shape[0]);
    try expectEqual(@as(usize, 3), @TypeOf(rows).shape[1]);
    try expectEqualSlices(f32, &.{ 3, 4, 5, 6, 7, 8 }, rows.data);

    a.data[3] = 100;
    try expectEqual(@as(f32, 100), rows.data[0]);
}

test "sumTo reduces what broadcasting expanded" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    try expectEqualSlices(f32, &.{ 5, 7, 9 }, (try sumTo(gpa, a, &.{3})).data);
    try expectEqualSlices(f32, &.{ 5, 7, 9 }, (try sumTo(gpa, a, &.{ 1, 3 })).data);
    try expectEqualSlices(f32, &.{ 6, 15 }, (try sumTo(gpa, a, &.{ 2, 1 })).data);
    try expectEqualSlices(f32, &.{21}, (try sumTo(gpa, a, &.{})).data);
    try expectEqualSlices(f32, a.data, (try sumTo(gpa, a, &.{ 2, 3 })).data);

    const b = try Tensor(f32, &.{ 2, 2, 2 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    try expectEqualSlices(f32, &.{ 4, 6, 12, 14 }, (try sumTo(gpa, b, &.{ 2, 1, 2 })).data);
    try expectEqualSlices(f32, &.{ 3, 7, 11, 15 }, (try sumTo(gpa, b, &.{ 2, 2, 1 })).data);
}

test "sumTo is the adjoint of broadcasting" {
    var arena = testArena();
    defer arena.deinit();
    const gpa = arena.allocator();

    const x = try Tensor(f32, &.{ 1, 3 }).fromSlice(gpa, &.{ 1, 2, 3 });
    const g = try Tensor(f32, &.{ 4, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    const bx = try add(gpa, x, try Tensor(f32, &.{ 4, 3 }).zeros(gpa));
    const lhs = (try reduce(gpa, try mul(gpa, bx, g), null, .sum)).data[0];
    const rhs = (try reduce(gpa, try mul(gpa, x, try sumTo(gpa, g, &.{ 1, 3 })), null, .sum)).data[0];
    try expectApproxEqAbs(lhs, rhs, 1e-4);
}

test "ops free everything they allocate" {
    const gpa = testing.allocator;
    const a = try Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit(gpa);

    const r1 = try add(gpa, a, a);
    defer r1.deinit(gpa);
    const r2 = try softmax(gpa, a);
    defer r2.deinit(gpa);
    const r3 = try sumTo(gpa, a, &.{3});
    defer r3.deinit(gpa);
    const r4 = try reduce(gpa, a, 0, .sum);
    defer r4.deinit(gpa);
    const r5 = try reduce(gpa, a, 1, .argmax);
    defer r5.deinit(gpa);
}
