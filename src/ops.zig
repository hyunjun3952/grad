const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

const Error = Allocator.Error;

fn requireFloat(comptime T: type, comptime what: []const u8) void {
    if (@typeInfo(T) != .float) {
        @compileError(what ++ " requires a floating point element type");
    }
}

fn prod(comptime dims: []const usize) usize {
    var n: usize = 1;
    for (dims) |d| n *= d;
    return n;
}

fn sameShape(comptime a: []const usize, comptime b: []const usize) bool {
    return std.mem.eql(usize, a, b);
}

fn shaped(comptime E: type, comptime dims: anytype) type {
    const final = dims;
    return Tensor(E, &final);
}

fn Broadcast(comptime rank: usize) type {
    return struct { dims: [rank]usize, a: [rank]usize, b: [rank]usize };
}

fn broadcast(comptime a: []const usize, comptime b: []const usize) Broadcast(@max(a.len, b.len)) {
    const rank = @max(a.len, b.len);
    var r: Broadcast(rank) = undefined;
    var acc_a: usize = 1;
    var acc_b: usize = 1;
    var i: usize = rank;
    while (i > 0) {
        i -= 1;
        const da = if (i < rank - a.len) 1 else a[i - (rank - a.len)];
        const db = if (i < rank - b.len) 1 else b[i - (rank - b.len)];
        if (da != db and da != 1 and db != 1) {
            @compileError(std.fmt.comptimePrint("cannot broadcast shapes {any} and {any}", .{ a, b }));
        }
        r.dims[i] = @max(da, db);
        r.a[i] = if (da == 1) 0 else acc_a;
        r.b[i] = if (db == 1) 0 else acc_b;
        acc_a *= da;
        acc_b *= db;
    }
    return r;
}

fn Walk(comptime dims: []const usize, comptime strides: []const [dims.len]usize) type {
    return struct {
        const outer = dims.len - 1;

        idx: [outer]usize = @splat(0),
        off: [strides.len]usize = @splat(0),

        fn next(self: *@This()) void {
            var d: usize = outer;
            while (d > 0) {
                d -= 1;
                self.idx[d] += 1;
                inline for (0..strides.len) |k| self.off[k] += strides[k][d];
                if (self.idx[d] < dims[d]) return;
                inline for (0..strides.len) |k| self.off[k] -= strides[k][d] * dims[d];
                self.idx[d] = 0;
            }
        }
    };
}

fn Accum(comptime T: type) type {
    return if (@bitSizeOf(T) < 64) f64 else T;
}

fn finish(comptime T: type, x: Accum(T), comptime op: Reduction, n: usize) T {
    const y = if (op == .mean) x / @as(Accum(T), @floatFromInt(n)) else x;
    return if (Accum(T) == T) y else @floatCast(y);
}

fn sumSlice(comptime T: type, xs: []const T) Accum(T) {
    var acc: [4]Accum(T) = @splat(0);
    var i: usize = 0;
    while (i + 4 <= xs.len) : (i += 4) {
        inline for (0..4) |j| acc[j] += xs[i + j];
    }
    var total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while (i < xs.len) : (i += 1) total += xs[i];
    return total;
}

fn BroadcastResult(comptime A: type, comptime B: type) type {
    if (A.Elem != B.Elem) {
        @compileError("element types differ");
    }
    const bc = broadcast(A.shape, B.shape);
    return shaped(A.Elem, bc.dims);
}

fn binary(gpa: Allocator, a: anytype, b: anytype, comptime f: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    const R = BroadcastResult(A, B);
    const out = try R.empty(gpa);

    if (comptime sameShape(A.shape, B.shape)) {
        for (out.data, a.data, b.data) |*o, x, y| o.* = f(x, y);
    } else if (comptime B.len == 1 and R.len == A.len) {
        const y = b.data[0];
        for (out.data, a.data) |*o, x| o.* = f(x, y);
    } else if (comptime A.len == 1 and R.len == B.len) {
        const x = a.data[0];
        for (out.data, b.data) |*o, y| o.* = f(x, y);
    } else {
        const bc = comptime broadcast(A.shape, B.shape);
        const last = R.rank - 1;
        const inner = R.shape[last];
        var w: Walk(R.shape, &.{ bc.a, bc.b }) = .{};
        var base: usize = 0;
        while (base < R.len) : (base += inner) {
            for (out.data[base..][0..inner], 0..) |*o, i| {
                o.* = f(a.data[w.off[0] + i * bc.a[last]], b.data[w.off[1] + i * bc.b[last]]);
            }
            w.next();
        }
    }
    return out;
}

fn unary(gpa: Allocator, a: anytype, comptime f: anytype) Error!@TypeOf(a) {
    const out = try @TypeOf(a).empty(gpa);
    for (out.data, a.data) |*o, x| o.* = f(x);
    return out;
}

pub fn add(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, struct {
        fn f(x: anytype, y: anytype) @TypeOf(x) {
            return x + y;
        }
    }.f);
}

pub fn sub(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, struct {
        fn f(x: anytype, y: anytype) @TypeOf(x) {
            return x - y;
        }
    }.f);
}

pub fn mul(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, struct {
        fn f(x: anytype, y: anytype) @TypeOf(x) {
            return x * y;
        }
    }.f);
}

pub fn div(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, struct {
        fn f(x: anytype, y: anytype) @TypeOf(x) {
            return x / y;
        }
    }.f);
}

pub fn neg(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, struct {
        fn f(x: anytype) @TypeOf(x) {
            return -x;
        }
    }.f);
}

pub fn relu(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, struct {
        fn f(x: anytype) @TypeOf(x) {
            return @max(x, 0);
        }
    }.f);
}

pub fn exp(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, struct {
        fn f(x: anytype) @TypeOf(x) {
            return @exp(x);
        }
    }.f);
}

pub fn log(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, struct {
        fn f(x: anytype) @TypeOf(x) {
            return @log(x);
        }
    }.f);
}

pub const Reduction = enum { sum, mean, argmax };

fn ReduceResult(comptime A: type, comptime axis: ?usize, comptime op: Reduction) type {
    const E = if (op == .argmax) usize else A.Elem;
    const x = axis orelse {
        if (op == .argmax) @compileError("reduce(.argmax) needs an axis");
        return Tensor(E, &.{});
    };
    if (x >= A.rank) {
        @compileError(std.fmt.comptimePrint("axis {d} out of range for rank {d}", .{ x, A.rank }));
    }
    var dims: [A.rank - 1]usize = undefined;
    var j: usize = 0;
    for (A.shape, 0..) |d, i| {
        if (i != x) {
            dims[j] = d;
            j += 1;
        }
    }
    return shaped(E, dims);
}

pub fn reduce(
    gpa: Allocator,
    a: anytype,
    comptime axis: ?usize,
    comptime op: Reduction,
) Error!ReduceResult(@TypeOf(a), axis, op) {
    const A = @TypeOf(a);
    const R = ReduceResult(A, axis, op);
    const outer = comptime if (axis) |x| prod(A.shape[0..x]) else 1;
    const mid = comptime if (axis) |x| A.shape[x] else A.len;
    const inner = comptime if (axis) |x| prod(A.shape[x + 1 ..]) else 1;

    const Acc = Accum(A.Elem);
    const wide = comptime op != .argmax and inner > 1 and Acc != A.Elem;
    const scratch = try gpa.alloc(Acc, if (wide) inner else 0);
    defer gpa.free(scratch);

    const out = try R.empty(gpa);
    for (0..outer) |o| {
        const src = a.data[o * mid * inner ..][0 .. mid * inner];
        const dst = out.data[o * inner ..][0..inner];
        switch (op) {
            .sum, .mean => {
                if (inner == 1) {
                    dst[0] = finish(A.Elem, sumSlice(A.Elem, src), op, mid);
                } else {
                    const acc: []Acc = if (wide) scratch else dst;
                    @memset(acc, 0);
                    for (0..mid) |m| {
                        for (acc, src[m * inner ..][0..inner]) |*d, s| d.* += s;
                    }
                    for (dst, acc) |*d, x| d.* = finish(A.Elem, x, op, mid);
                }
            },
            .argmax => {
                for (dst, 0..) |*d, i| {
                    var best = src[i];
                    var best_idx: usize = 0;
                    for (1..mid) |m| {
                        const v = src[m * inner + i];
                        if (v > best) {
                            best = v;
                            best_idx = m;
                        }
                    }
                    d.* = best_idx;
                }
            },
        }
    }
    return out;
}

fn softmaxRows(gpa: Allocator, a: anytype, comptime take_log: bool) Error!@TypeOf(a) {
    const A = @TypeOf(a);
    comptime {
        requireFloat(A.Elem, "softmax");
        if (A.rank == 0) @compileError("softmax requires rank >= 1");
    }
    const cols = A.shape[A.rank - 1];
    const rows = A.len / cols;

    const out = try A.empty(gpa);
    for (0..rows) |r| {
        const src = a.data[r * cols ..][0..cols];
        const dst = out.data[r * cols ..][0..cols];

        var mx = src[0];
        for (src[1..]) |x| mx = @max(mx, x);

        var total: A.Elem = 0;
        for (src, dst) |x, *d| {
            const e = @exp(x - mx);
            d.* = e;
            total += e;
        }
        if (take_log) {
            const shift = mx + @log(total);
            for (src, dst) |x, *d| d.* = x - shift;
        } else {
            for (dst) |*d| d.* /= total;
        }
    }
    return out;
}

pub fn softmax(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return softmaxRows(gpa, a, false);
}

pub fn logSoftmax(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return softmaxRows(gpa, a, true);
}

fn RowsViewResult(comptime A: type, comptime count: usize) type {
    if (A.rank == 0) @compileError("rowsView requires rank >= 1");
    if (count > A.shape[0]) @compileError("rowsView: count exceeds the first dimension");
    var dims: [A.rank]usize = undefined;
    dims[0] = count;
    for (1..A.rank) |i| dims[i] = A.shape[i];
    return shaped(A.Elem, dims);
}

pub fn rowsView(a: anytype, start: usize, comptime count: usize) RowsViewResult(@TypeOf(a), count) {
    const A = @TypeOf(a);
    const row_len = comptime A.len / A.shape[0];
    std.debug.assert(start + count <= A.shape[0]);
    return .{ .data = a.data[start * row_len ..][0 .. count * row_len] };
}

pub fn sumTo(gpa: Allocator, a: anytype, comptime target: []const usize) Error!Tensor(@TypeOf(a).Elem, target) {
    const A = @TypeOf(a);
    const R = Tensor(A.Elem, target);
    const bc = comptime broadcast(target, A.shape);
    comptime {
        if (!sameShape(&bc.dims, A.shape)) {
            @compileError(std.fmt.comptimePrint("sumTo: cannot reduce {any} to {any}", .{ A.shape, target }));
        }
    }
    if (comptime sameShape(A.shape, target)) return a.clone(gpa);

    const last = A.rank - 1;
    const inner = A.shape[last];

    const Acc = Accum(A.Elem);
    const scratch = try gpa.alloc(Acc, if (Acc == A.Elem) 0 else R.len);
    defer gpa.free(scratch);

    const out = try R.zeros(gpa);
    const acc: []Acc = if (Acc == A.Elem) out.data else scratch;
    if (Acc != A.Elem) @memset(acc, 0);

    var w: Walk(A.shape, &.{bc.a}) = .{};
    var base: usize = 0;
    while (base < A.len) : (base += inner) {
        const row = a.data[base..][0..inner];
        if (comptime bc.a[last] == 0) {
            acc[w.off[0]] += sumSlice(A.Elem, row);
        } else {
            for (row, 0..) |x, i| acc[w.off[0] + i * bc.a[last]] += x;
        }
        w.next();
    }
    if (Acc != A.Elem) {
        for (out.data, acc) |*o, x| o.* = finish(A.Elem, x, .sum, 1);
    }
    return out;
}

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
