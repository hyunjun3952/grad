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

pub fn broadcast(comptime a: []const usize, comptime b: []const usize) Broadcast(@max(a.len, b.len)) {
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

pub fn BroadcastResult(comptime A: type, comptime B: type) type {
    if (A.Elem != B.Elem) {
        @compileError("element types differ");
    }
    const bc = broadcast(A.shape, B.shape);
    return shaped(A.Elem, bc.dims);
}

const BinaryOp = enum { add, sub, mul, div };
const UnaryOp = enum { neg, relu, exp, log };

fn applyBinary(comptime op: BinaryOp, x: anytype, y: anytype) @TypeOf(x) {
    return switch (op) {
        .add => x + y,
        .sub => x - y,
        .mul => x * y,
        .div => x / y,
    };
}

fn binary(gpa: Allocator, a: anytype, b: anytype, comptime op: BinaryOp) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    const R = BroadcastResult(A, B);
    const out = try R.empty(gpa);

    if (comptime sameShape(A.shape, B.shape)) {
        for (out.data, a.data, b.data) |*o, x, y| o.* = applyBinary(op, x, y);
    } else if (comptime B.len == 1 and R.len == A.len) {
        const y = b.data[0];
        for (out.data, a.data) |*o, x| o.* = applyBinary(op, x, y);
    } else if (comptime A.len == 1 and R.len == B.len) {
        const x = a.data[0];
        for (out.data, b.data) |*o, y| o.* = applyBinary(op, x, y);
    } else {
        const bc = comptime broadcast(A.shape, B.shape);
        const last = R.rank - 1;
        const inner = R.shape[last];
        var w: Walk(R.shape, &.{ bc.a, bc.b }) = .{};
        var base: usize = 0;
        while (base < R.len) : (base += inner) {
            for (out.data[base..][0..inner], 0..) |*o, i| {
                o.* = applyBinary(op, a.data[w.off[0] + i * bc.a[last]], b.data[w.off[1] + i * bc.b[last]]);
            }
            w.next();
        }
    }
    return out;
}

fn applyUnary(comptime op: UnaryOp, x: anytype) @TypeOf(x) {
    return switch (op) {
        .neg => -x,
        .relu => @max(x, 0),
        .exp => @exp(x),
        .log => @log(x),
    };
}

fn unary(gpa: Allocator, a: anytype, comptime op: UnaryOp) Error!@TypeOf(a) {
    const out = try @TypeOf(a).empty(gpa);
    for (out.data, a.data) |*o, x| o.* = applyUnary(op, x);
    return out;
}

pub fn add(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, .add);
}

pub fn sub(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, .sub);
}

pub fn mul(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, .mul);
}

pub fn div(gpa: Allocator, a: anytype, b: anytype) Error!BroadcastResult(@TypeOf(a), @TypeOf(b)) {
    return binary(gpa, a, b, .div);
}

pub fn neg(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, .neg);
}

pub fn relu(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, .relu);
}

pub fn exp(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, .exp);
}

pub fn log(gpa: Allocator, a: anytype) Error!@TypeOf(a) {
    return unary(gpa, a, .log);
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
