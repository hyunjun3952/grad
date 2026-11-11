const std = @import("std");
const kernels = @import("../cpu/kernels.zig");
const ga = @import("../gpu/args.zig");

// ***** optimizers *****

// NOTE: the device kernels take a table of raw addresses so that one launch updates many tensors.
// A table holds at most max_tensors entries, so a model with more tensors is updated in several launches, one per chunk.
fn countOf(comptime P: type) usize {
    return switch (@typeInfo(P)) {
        .array => |array| array.len,
        else => @typeInfo(P).@"struct".field_types.len,
    };
}

fn chunkCount(comptime P: type) usize {
    return (countOf(P) + ga.max_tensors - 1) / ga.max_tensors;
}

fn tensorTable(params: anytype, comptime chunk: usize) ga.TensorTable {
    const first = chunk * ga.max_tensors;
    var table = std.mem.zeroes(ga.TensorTable);
    table.count = @min(ga.max_tensors, countOf(@TypeOf(params)) - first);
    inline for (params, 0..) |p, i| {
        if (i >= first and i < first + ga.max_tensors) {
            table.len[i - first] = @intCast(p.value.data.len);
            table.w[i - first] = @intFromPtr(p.value.data.ptr);
            table.g[i - first] = @intFromPtr(p.grad.data.ptr);
        }
    }
    return table;
}

/// Sets the gradient of every parameter to zero.
pub fn zeroGrad(par: kernels.Parallel, params: anytype) !void {
    if (par.gpu) |dev| {
        inline for (0..comptime chunkCount(@TypeOf(params))) |chunk| try dev.zeroGrads(tensorTable(params, chunk));
        return;
    }
    inline for (params) |p| {
        const G = @TypeOf(p.grad.data);
        const body = struct {
            fn run(g: G, lo: usize, hi: usize) void {
                @memset(g[lo..hi], 0);
            }
        }.run;
        kernels.parallelFor(par, p.grad.data.len, 1 << 15, p.grad.data, body);
    }
}

/// Takes one plain gradient descent step, w -= lr * grad, on every parameter.
pub fn sgd(par: kernels.Parallel, lr: f32, params: anytype) !void {
    if (par.gpu) |dev| {
        inline for (0..comptime chunkCount(@TypeOf(params))) |chunk| try dev.sgd(tensorTable(params, chunk), lr);
        return;
    }
    inline for (params) |p| {
        const E = @TypeOf(p.value).Elem;
        const Ctx = struct { w: []E, g: []const E, lr: E };
        const body = struct {
            fn run(c: Ctx, lo: usize, hi: usize) void {
                for (c.w[lo..hi], c.g[lo..hi]) |*w, g| w.* -= c.lr * g;
            }
        }.run;
        kernels.parallelFor(par, p.value.data.len, 1 << 14, Ctx{ .w = p.value.data, .g = p.grad.data, .lr = lr }, body);
    }
}

/// Returns the learning rate for a step: a linear ramp over the first warmup steps, then a cosine decay from base to zero at total.
pub fn cosineLr(base: f32, step: usize, total: usize, warmup: usize) f32 {
    if (step < warmup) return base * @as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(warmup));
    const progress = @as(f32, @floatFromInt(step - warmup)) / @as(f32, @floatFromInt(total - warmup));
    return 0.5 * base * (1 + @cos(std.math.pi * progress));
}

/// Adam with optional decoupled weight decay. The moments m and v live in one buffer each, laid out in parameter order.
pub const Adam = struct {
    gpa: std.mem.Allocator,
    on_gpu: bool,
    m: []f32,
    v: []f32,
    t: u32 = 0,
    lr: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0,

    /// Allocates zeroed moments for params, on the device when par has one.
    pub fn init(gpa: std.mem.Allocator, par: kernels.Parallel, lr: f32, params: anytype) !Adam {
        var total: usize = 0;
        inline for (params) |p| total += p.value.data.len;
        if (par.gpu) |dev| {
            const m = try dev.alloc(f32, total);
            const v = try dev.alloc(f32, total);
            try dev.fill(m, 0);
            try dev.fill(v, 0);
            return .{ .gpa = gpa, .on_gpu = true, .m = m, .v = v, .lr = lr };
        }
        const m = try gpa.alloc(f32, total);
        errdefer gpa.free(m);
        const v = try gpa.alloc(f32, total);
        @memset(m, 0);
        @memset(v, 0);
        return .{ .gpa = gpa, .on_gpu = false, .m = m, .v = v, .lr = lr };
    }

    /// Returns the two moment buffers so that a checkpoint can store them.
    pub fn state(self: Adam) struct { []f32, []f32 } {
        return .{ self.m, self.v };
    }

    /// Frees the host buffers; device buffers belong to the Gpu.
    pub fn deinit(self: Adam) void {
        if (self.on_gpu) return;
        self.gpa.free(self.m);
        self.gpa.free(self.v);
    }

    /// Applies one update to params using their gradients.
    /// NOTE: the bias corrections are folded into the step size and the second moment scale, so the kernels need no powers.
    pub fn step(self: *Adam, par: kernels.Parallel, params: anytype) !void {
        self.t += 1;
        const t: f64 = @floatFromInt(self.t);
        const c1 = 1 - std.math.pow(f64, self.beta1, t);
        const c2 = 1 - std.math.pow(f64, self.beta2, t);

        if (par.gpu) |dev| {
            const hyper: ga.AdamArgs = .{
                .beta1 = self.beta1,
                .beta2 = self.beta2,
                .step = @floatCast(self.lr / c1),
                .inv_sqrt_c2 = @floatCast(1 / @sqrt(c2)),
                .eps = self.eps,
                .decay = self.lr * self.weight_decay,
            };
            inline for (0..comptime chunkCount(@TypeOf(params))) |chunk| {
                const first = chunk * ga.max_tensors;
                var table = tensorTable(params, chunk);
                var at: usize = 0;
                inline for (params, 0..) |p, i| {
                    if (i >= first and i < first + ga.max_tensors) {
                        table.m[i - first] = @intFromPtr(self.m.ptr) + @sizeOf(f32) * at;
                        table.v[i - first] = @intFromPtr(self.v.ptr) + @sizeOf(f32) * at;
                    }
                    at += p.value.data.len;
                }
                try dev.adam(table, hyper);
            }
            return;
        }
        var off: usize = 0;
        inline for (params) |p| {
            const len = p.value.data.len;
            const Ctx = struct { w: []f32, g: []const f32, m: []f32, v: []f32, b1: f32, b2: f32, step: f32, inv_sqrt_c2: f32, eps: f32, decay: f32 };
            const body = struct {
                fn run(c: Ctx, lo: usize, hi: usize) void {
                    const lanes = std.simd.suggestVectorLength(f32) orelse 1;
                    var i = lo;
                    while (i + lanes <= hi) : (i += lanes) update(lanes, c, i);
                    while (i < hi) : (i += 1) update(1, c, i);
                }

                inline fn update(comptime n: usize, c: Ctx, i: usize) void {
                    const V = @Vector(n, f32);
                    const g: V = c.g[i..][0..n].*;
                    var m: V = c.m[i..][0..n].*;
                    var v: V = c.v[i..][0..n].*;
                    const b1: V = @splat(c.b1);
                    const b2: V = @splat(c.b2);
                    const one: V = @splat(1);
                    m = b1 * m + (one - b1) * g;
                    v = b2 * v + (one - b2) * g * g;
                    const scale: V = @splat(c.step);
                    const inv_sqrt_c2: V = @splat(c.inv_sqrt_c2);
                    const eps: V = @splat(c.eps);
                    const w: V = c.w[i..][0..n].*;
                    const decay: V = @splat(c.decay);
                    c.w[i..][0..n].* = w - scale * m / (@sqrt(v) * inv_sqrt_c2 + eps) - decay * w;
                    c.m[i..][0..n].* = m;
                    c.v[i..][0..n].* = v;
                }
            }.run;
            const ctx: Ctx = .{
                .w = p.value.data,
                .g = p.grad.data,
                .m = self.m[off..][0..len],
                .v = self.v[off..][0..len],
                .b1 = self.beta1,
                .b2 = self.beta2,
                .step = @floatCast(self.lr / c1),
                .inv_sqrt_c2 = @floatCast(1 / @sqrt(c2)),
                .eps = self.eps,
                .decay = self.lr * self.weight_decay,
            };
            kernels.parallelFor(par, len, 1 << 14, ctx, body);
            off += len;
        }
    }
};

/// An exponential moving average of the weights. The shadow copy is kept in one buffer; swap exchanges it with the live weights for evaluation.
pub const Ema = struct {
    gpa: std.mem.Allocator,
    on_gpu: bool,
    values: []f32,
    live: [][]f32,
    shadow: [][]f32,
    decay: f32 = 0,

    /// Starts the shadow as a copy of the given weight slices. decay is set by the caller before the first update.
    pub fn init(gpa: std.mem.Allocator, par: kernels.Parallel, slices: anytype) !Ema {
        var total: usize = 0;
        inline for (slices) |s| total += s.len;
        const live = try gpa.alloc([]f32, slices.len);
        errdefer gpa.free(live);
        const shadow = try gpa.alloc([]f32, slices.len);
        errdefer gpa.free(shadow);
        const on_gpu = par.gpu != null;
        const values = if (par.gpu) |dev| try dev.alloc(f32, total) else try gpa.alloc(f32, total);
        errdefer if (!on_gpu) gpa.free(values);
        var at: usize = 0;
        inline for (slices, 0..) |s, i| {
            live[i] = s;
            shadow[i] = values[at..][0..s.len];
            at += s.len;
        }
        if (par.gpu) |dev| {
            const host = try gpa.alloc(f32, total);
            defer gpa.free(host);
            var copied: usize = 0;
            for (live) |s| {
                try dev.download(host[copied..][0..s.len], s);
                copied += s.len;
            }
            try dev.upload(values, host);
        } else {
            for (live, shadow) |s, d| @memcpy(d, s);
        }
        return .{ .gpa = gpa, .on_gpu = on_gpu, .values = values, .live = live, .shadow = shadow };
    }

    pub fn deinit(self: Ema) void {
        if (!self.on_gpu) self.gpa.free(self.values);
        self.gpa.free(self.live);
        self.gpa.free(self.shadow);
    }

    /// Returns the shadow buffer so that a checkpoint can store it.
    pub fn state(self: Ema) struct { []f32 } {
        return .{self.values};
    }

    /// Moves the shadow towards the live weights.
    /// NOTE: the decay is capped at (step + 1) / (step + 10) so that early steps do not keep the random initial weights.
    pub fn update(self: Ema, par: kernels.Parallel, step: usize) !void {
        const warm = @as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(step + 10));
        try self.apply(par, .average, @min(self.decay, warm));
    }

    /// Exchanges the live weights with the shadow; calling it twice restores the original state.
    pub fn swap(self: Ema, par: kernels.Parallel) !void {
        try self.apply(par, .swap, 0);
    }

    fn apply(self: Ema, par: kernels.Parallel, comptime op: enum { average, swap }, decay: f32) !void {
        if (par.gpu) |dev| {
            var first: usize = 0;
            while (first < self.live.len) : (first += ga.max_tensors) {
                var table = std.mem.zeroes(ga.TensorTable);
                const count = @min(ga.max_tensors, self.live.len - first);
                table.count = @intCast(count);
                for (0..count) |i| {
                    table.len[i] = @intCast(self.live[first + i].len);
                    table.w[i] = @intFromPtr(self.live[first + i].ptr);
                    table.g[i] = @intFromPtr(self.shadow[first + i].ptr);
                }
                if (op == .average) try dev.ema(table, decay) else try dev.swapWeights(table);
            }
            return;
        }
        const Ctx = struct { live: []f32, shadow: []f32, decay: f32 };
        const body = struct {
            fn run(c: Ctx, lo: usize, hi: usize) void {
                for (c.live[lo..hi], c.shadow[lo..hi]) |*w, *s| {
                    if (op == .average) {
                        s.* = c.decay * s.* + (1 - c.decay) * w.*;
                    } else {
                        const kept = w.*;
                        w.* = s.*;
                        s.* = kept;
                    }
                }
            }
        }.run;
        for (self.live, self.shadow) |w, s| kernels.parallelFor(par, w.len, 1 << 14, Ctx{ .live = w, .shadow = s, .decay = decay }, body);
    }
};
