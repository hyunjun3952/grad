const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const cuda = @import("cuda.zig");
const conv = @import("conv.zig");
const ga = @import("args.zig");

const ptx = @embedFile("kernels_ptx");

pub const Error = Allocator.Error || error{ DriverNotFound, MissingSymbol, CudaFailed };

pub const BinaryOp = enum { add, mul };

pub const Augment = struct { max_shift: u32 = 0, seed: u64 = 0 };

pub const MatmulOpts = struct { trans_a: bool = false, trans_b: bool = false, accumulate: bool = false };

pub const MatmulShape = struct { m: usize, n: usize, k: usize };

pub const BroadcastPlan = struct { dims: [4]u32, a_strides: [4]u32, b_strides: [4]u32 };

pub const ReducePlan = struct { dims: [4]u32, strides: [4]u32, mask: u32 };

const Kernel = enum {
    sgemmNN,
    sgemmNT,
    sgemmTN,
    sgemmTallNN,
    sgemmTallNT,
    sgemmTallTN,
    sgemmSmallNN,
    sgemmSmallNT,
    sgemmSmallTN,
    elementwise,
    relu,
    reluBackward,
    addInPlace,
    sumTo,
    crossEntropyRows,
    crossEntropyMean,
    crossEntropyBackward,
    zeroGrads,
    sgd,
    adam,
    gather,
    im2col,
    col2im,
    maxPool,
    maxPoolBackward,
    avgPool,
    avgPoolBackward,
    dropout,
    columnReduce,
    batchNormFinish,
    batchNormApply,
    batchNormGradFinish,
    batchNormGradInput,
};
const kernel_count = @typeInfo(Kernel).@"enum".field_names.len;

fn address(slice: anytype) u64 {
    return @intFromPtr(slice.ptr);
}

fn toU32(x: usize) u32 {
    return @intCast(x);
}

fn ceilDiv(x: usize, y: usize) u32 {
    return @intCast((x + y - 1) / y);
}

const slot_fill = kernel_count;
const slot_upload = kernel_count + 1;
const slot_download = kernel_count + 2;
const slot_count = kernel_count + 3;

fn slotName(slot: usize) []const u8 {
    return switch (slot) {
        slot_fill => "fill (memset)",
        slot_upload => "upload",
        slot_download => "download",
        else => @tagName(@as(Kernel, @fromBackingInt(@intCast(slot)))),
    };
}

const Profile = struct { io: Io, ns: [slot_count]i96 = @splat(0), calls: [slot_count]u32 = @splat(0) };

pub const Gpu = struct {
    profile: ?Profile = null,
    gpa: Allocator,
    cu: cuda.Cuda,
    fns: [kernel_count]cuda.Handle,
    owned: std.ArrayList(cuda.DevicePtr),
    step_base: usize,
    step_size: usize,
    step_used: usize,
    zero_base: usize,
    zero_size: usize,
    zero_used: usize,

    pub fn init(gpa: Allocator, step_bytes: usize) Error!*Gpu {
        const self = try gpa.create(Gpu);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.profile = null;
        self.owned = .empty;
        self.cu = try cuda.Cuda.open();
        errdefer self.cu.close();

        const module = try self.cu.loadModule(gpa, ptx);
        inline for (0..kernel_count) |i| {
            self.fns[i] = try self.cu.getFunction(module, @tagName(@as(Kernel, @fromBackingInt(@intCast(i)))));
        }
        self.step_base = try self.cu.alloc(step_bytes);
        self.step_size = step_bytes;
        self.step_used = 0;
        self.zero_size = step_bytes / 4;
        self.zero_base = try self.cu.alloc(self.zero_size);
        self.zero_used = 0;
        try self.cu.memsetD32(self.zero_base, 0, self.zero_size / 4);
        return self;
    }

    pub fn deinit(self: *Gpu) void {
        for (self.owned.items) |ptr| self.cu.free(ptr);
        self.cu.free(self.step_base);
        self.cu.free(self.zero_base);
        self.owned.deinit(self.gpa);
        self.cu.close();
        self.gpa.destroy(self);
    }

    pub fn alloc(self: *Gpu, comptime T: type, n: usize) Error![]T {
        const ptr = try self.cu.alloc(@sizeOf(T) * n);
        errdefer self.cu.free(ptr);
        try self.owned.append(self.gpa, ptr);
        return @as([*]T, @ptrFromInt(ptr))[0..n];
    }

    pub fn scratch(self: *Gpu, comptime T: type, n: usize) Error![]T {
        const at = std.mem.alignForward(usize, self.step_used, 256);
        const bytes = @sizeOf(T) * n;
        if (at + bytes > self.step_size) return error.OutOfMemory;
        self.step_used = at + bytes;
        return @as([*]T, @ptrFromInt(self.step_base + at))[0..n];
    }

    pub fn scratchZeroed(self: *Gpu, comptime T: type, n: usize) Error![]T {
        const at = std.mem.alignForward(usize, self.zero_used, 256);
        const bytes = @sizeOf(T) * n;
        if (at + bytes > self.zero_size) return error.OutOfMemory;
        self.zero_used = at + bytes;
        return @as([*]T, @ptrFromInt(self.zero_base + at))[0..n];
    }

    pub fn resetStep(self: *Gpu) Error!void {
        self.step_used = 0;
        if (self.zero_used == 0) return;
        const started = try self.timerStart();
        try self.cu.memsetD32(self.zero_base, 0, std.mem.alignForward(usize, self.zero_used, 4) / 4);
        try self.timerStop(slot_fill, started);
        self.zero_used = 0;
    }

    pub fn startProfile(self: *Gpu, io: Io) void {
        self.profile = .{ .io = io };
    }

    fn timerStart(self: *Gpu) Error!i96 {
        const p = self.profile orelse return 0;
        try self.cu.sync();
        return Io.Clock.awake.now(p.io).nanoseconds;
    }

    fn timerStop(self: *Gpu, slot: usize, started: i96) Error!void {
        const p = &(self.profile orelse return);
        try self.cu.sync();
        p.ns[slot] += Io.Clock.awake.now(p.io).nanoseconds - started;
        p.calls[slot] += 1;
    }

    pub fn printProfile(self: *Gpu, out: *Io.Writer) !void {
        const p = self.profile orelse return;
        var order: [slot_count]usize = undefined;
        for (&order, 0..) |*o, i| o.* = i;
        std.sort.pdq(usize, &order, &p.ns, struct {
            fn more(ns: *const [slot_count]i96, a: usize, b: usize) bool {
                return ns[a] > ns[b];
            }
        }.more);
        var total: i96 = 0;
        for (p.ns) |ns| total += ns;
        try out.print("{s:<22}{s:>8}{s:>12}{s:>10}{s:>8}\n", .{ "gpu op", "calls", "total ms", "us/call", "share" });
        for (order) |slot| {
            if (p.calls[slot] == 0) continue;
            const ms = @as(f64, @floatFromInt(p.ns[slot])) / 1e6;
            try out.print("{s:<22}{d:>8}{d:>12.1}{d:>10.1}{d:>7.1}%\n", .{ slotName(slot), p.calls[slot], ms, ms * 1000 / @as(f64, @floatFromInt(p.calls[slot])), 100 * @as(f64, @floatFromInt(p.ns[slot])) / @as(f64, @floatFromInt(total)) });
        }
    }

    pub fn upload(self: *Gpu, dst: anytype, src: anytype) Error!void {
        std.debug.assert(dst.len == src.len);
        const started = try self.timerStart();
        try self.cu.upload(@intFromPtr(dst.ptr), src);
        try self.timerStop(slot_upload, started);
    }

    pub fn download(self: *Gpu, dst: anytype, src: anytype) Error!void {
        std.debug.assert(dst.len == src.len);
        const started = try self.timerStart();
        try self.cu.download(dst, @intFromPtr(src.ptr));
        try self.timerStop(slot_download, started);
    }

    pub fn fill(self: *Gpu, dst: []f32, value: f32) Error!void {
        const started = try self.timerStart();
        try self.cu.memsetD32(@intFromPtr(dst.ptr), @bitCast(value), dst.len);
        try self.timerStop(slot_fill, started);
    }

    pub fn scalar(self: *Gpu, src: []const f32) Error!f32 {
        var out: [1]f32 = undefined;
        try self.download(&out, src[0..1]);
        return out[0];
    }

    fn launch(self: *Gpu, kernel: Kernel, grid: [3]u32, args: anytype) Error!void {
        if (grid[0] == 0 or grid[1] == 0) return;
        var copy = args;
        var pointers: [copy.len]?*anyopaque = undefined;
        inline for (0..copy.len) |i| {
            if (@sizeOf(@TypeOf(copy[i])) < 4) @compileError("kernel arguments must be at least 32 bits wide; widen flags with @as(u32, ...)");
            pointers[i] = &copy[i];
        }
        const started = try self.timerStart();
        try self.cu.launch(self.fns[@backingInt(kernel)], grid, .{ 256, 1, 1 }, &pointers);
        try self.timerStop(@backingInt(kernel), started);
    }

    fn launchFlat(self: *Gpu, kernel: Kernel, items: usize, args: anytype) Error!void {
        try self.launch(kernel, .{ ceilDiv(items, 256), 1, 1 }, args);
    }

    pub fn matmul(self: *Gpu, a: []const f32, b: []const f32, out: []f32, shape: MatmulShape, opts: MatmulOpts) Error!void {
        const m = shape.m;
        const n = shape.n;
        const k = shape.k;
        const trans_a = opts.trans_a;
        const trans_b = opts.trans_b;
        const accumulate = opts.accumulate;
        std.debug.assert(!(trans_a and trans_b));
        const tile: usize = if (n <= 32 and m <= 64) 2 else if (n <= 32) 1 else 0;
        const transpose: usize = if (trans_a) 2 else if (trans_b) 1 else 0;
        const kernel: Kernel = @fromBackingInt(@backingInt(Kernel.sgemmNN) + @as(u5, @intCast(tile * 3 + transpose)));
        const tile_m: usize = if (tile == 2) 32 else 128;
        const tile_n: usize = if (tile == 0) 128 else 32;
        const blocks = @as(usize, ceilDiv(n, tile_n)) * ceilDiv(m, tile_m);
        const splits = @max(1, @min(k / 256, (128 + blocks - 1) / blocks));
        const chunk = std.mem.alignForward(usize, (k + splits - 1) / splits, 8);
        if (splits > 1 and !accumulate) try self.fill(out, 0);
        const kernel_shape: ga.GemmShape = .{
            .m = toU32(m),
            .n = toU32(n),
            .k = toU32(k),
            .accumulate = @intFromBool(accumulate),
            .chunk = toU32(chunk),
        };
        try self.launch(kernel, .{ ceilDiv(n, tile_n), ceilDiv(m, tile_m), ceilDiv(k, chunk) }, .{ address(a), address(b), address(out), kernel_shape });
    }
    pub fn elementwise(self: *Gpu, op: BinaryOp, a: []const f32, b: []const f32, out: []f32, plan: BroadcastPlan) Error!void {
        const args: ga.ElementwiseArgs = .{ .len = toU32(out.len), .multiply = @intFromBool(op == .mul), .dims = plan.dims, .a_strides = plan.a_strides, .b_strides = plan.b_strides };
        try self.launchFlat(.elementwise, out.len, .{ address(a), address(b), address(out), args });
    }
    pub fn relu(self: *Gpu, x: []const f32, out: []f32) Error!void {
        try self.launchFlat(.relu, x.len, .{ address(x), address(out), toU32(x.len) });
    }

    pub fn reluBackward(self: *Gpu, x: []const f32, g_out: []const f32, g_in: []f32) Error!void {
        try self.launchFlat(.reluBackward, x.len, .{ address(x), address(g_out), address(g_in), toU32(x.len) });
    }

    pub fn addInPlace(self: *Gpu, dst: []f32, src: []const f32) Error!void {
        try self.launchFlat(.addInPlace, dst.len, .{ address(dst), address(src), toU32(dst.len) });
    }

    pub fn sumTo(self: *Gpu, in: []const f32, out: []f32, plan: ReducePlan, accumulate: bool) Error!void {
        if (plan.mask == 0b0111 and plan.strides[3] == 1 and plan.dims[3] == out.len) {
            if (!accumulate) try self.fill(out, 0);
            return self.columnSums(in, out);
        }
        const args: ga.SumArgs = .{ .out_len = toU32(out.len), .reduce_count = toU32(in.len / out.len), .mask = plan.mask, .accumulate = @intFromBool(accumulate), .dims = plan.dims, .strides = plan.strides };
        try self.launch(.sumTo, .{ toU32(out.len), 1, 1 }, .{ address(in), address(out), args });
    }
    pub fn crossEntropy(self: *Gpu, logits: []const f32, logp: []f32, labels: []const u8, loss: []f32, cols: usize) Error!void {
        const rows = labels.len;
        const row_loss = try self.scratch(f32, rows);
        try self.launchFlat(.crossEntropyRows, rows, .{ address(logits), address(logp), address(labels), address(row_loss), toU32(rows), toU32(cols) });
        try self.launchFlat(.crossEntropyMean, 1, .{ address(row_loss), address(loss), toU32(rows) });
    }

    pub fn crossEntropyBackward(self: *Gpu, logp: []const f32, labels: []const u8, grad: []f32, loss_grad: []const f32, cols: usize) Error!void {
        const rows = labels.len;
        try self.launchFlat(.crossEntropyBackward, rows * cols, .{ address(logp), address(labels), address(grad), address(loss_grad), toU32(rows), toU32(cols) });
    }

    pub fn convKernel(self: *Gpu, comptime op: conv.Op, slices: anytype, d: conv.Dims) Error!void {
        const len = conv.workItems(op, d);
        const types = comptime blk: {
            var t: [slices.len + 2]type = undefined;
            for (&t, 0..) |*e, i| e.* = if (i < slices.len) u64 else if (i == slices.len) u32 else conv.Dims;
            break :blk t;
        };
        var args: @Tuple(&types) = undefined;
        inline for (slices, 0..) |s, i| args[i] = address(s);
        args[slices.len] = toU32(len);
        args[slices.len + 1] = d;
        try self.launchFlat(@field(Kernel, @tagName(op)), len, args);
    }
    const ColumnKind = enum(u32) { sum, squared_deviation, gradient_sums };

    fn columnReduce(self: *Gpu, a: []const f32, b: []const f32, mean: []const f32, inv_std: []const f32, out: []f32, kind: ColumnKind, scale: f32, channels: usize) Error!void {
        const rows = a.len / channels;
        const blocks: u32 = @min(256, ceilDiv(rows, 8 * 16));
        try self.launch(.columnReduce, .{ blocks, ceilDiv(channels, 32), 1 }, .{
            address(a),  address(b),      address(mean), address(inv_std),  address(out),
            toU32(rows), toU32(channels), blocks * 8,    @backingInt(kind), scale,
        });
    }

    pub fn columnSums(self: *Gpu, in: []const f32, out: []f32) Error!void {
        try self.columnReduce(in, in, in[0..out.len], in[0..out.len], out, .sum, 1, out.len);
    }

    pub fn batchNormStats(self: *Gpu, x: []const f32, mean: []f32, inv_std: []f32, running_mean: []f32, running_var: []f32, training: bool) Error!void {
        const c = mean.len;
        const sums = try self.scratchZeroed(f32, 2 * c);
        if (training) {
            try self.columnReduce(x, x, sums[0..c], sums[0..c], sums[0..c], .sum, 1, c);
            try self.columnReduce(x, x, sums[0..c], sums[0..c], sums[c..], .squared_deviation, 1 / @as(f32, @floatFromInt(x.len / c)), c);
        }
        try self.launchFlat(.batchNormFinish, c, .{ address(sums), address(mean), address(inv_std), address(running_mean), address(running_var), toU32(x.len / c), toU32(c), @as(u32, @intFromBool(training)) });
    }
    pub fn batchNormApply(self: *Gpu, x: []const f32, y: []f32, mean: []const f32, inv_std: []const f32, gamma: []const f32, beta: []const f32) Error!void {
        try self.launchFlat(.batchNormApply, x.len, .{ address(x), address(y), toU32(x.len), toU32(mean.len), address(mean), address(inv_std), address(gamma), address(beta) });
    }

    pub fn batchNormGradSums(self: *Gpu, x: []const f32, dy: []const f32, mean: []const f32, inv_std: []const f32, sums: []f32, gamma_grad: []f32, beta_grad: []f32) Error!void {
        try self.columnReduce(dy, x, mean, inv_std, sums, .gradient_sums, 1, mean.len);
        try self.launchFlat(.batchNormGradFinish, mean.len, .{ address(sums), address(gamma_grad), address(beta_grad), toU32(mean.len) });
    }
    pub fn batchNormGradInput(self: *Gpu, x: []const f32, dy: []const f32, dx: []f32, mean: []const f32, inv_std: []const f32, gamma: []const f32, sums: []const f32) Error!void {
        try self.launchFlat(.batchNormGradInput, x.len, .{ address(x), address(dy), address(dx), toU32(x.len), toU32(mean.len), address(mean), address(inv_std), address(gamma), address(sums) });
    }

    pub fn dropout(self: *Gpu, src: []const f32, dst: []f32, seed: u64, keep: f32, accumulate: bool) Error!void {
        try self.launchFlat(.dropout, src.len, .{ address(src), address(dst), toU32(src.len), seed, keep, @as(u32, @intFromBool(accumulate)) });
    }

    fn launchTable(self: *Gpu, kernel: Kernel, table: ga.TensorTable, args: anytype) Error!void {
        var total: usize = 0;
        for (table.len[0..table.count]) |len| total += len;
        try self.launchFlat(kernel, total, .{table} ++ args);
    }

    pub fn zeroGrads(self: *Gpu, table: ga.TensorTable) Error!void {
        try self.launchTable(.zeroGrads, table, .{});
    }

    pub fn sgd(self: *Gpu, table: ga.TensorTable, lr: f32) Error!void {
        try self.launchTable(.sgd, table, .{lr});
    }

    pub fn adam(self: *Gpu, table: ga.TensorTable, hp: ga.AdamArgs) Error!void {
        try self.launchTable(.adam, table, .{hp});
    }
    pub fn gather(self: *Gpu, images: []const f32, labels_src: []const u8, order: []const u32, batch: []f32, labels_out: []u8, start: usize, augment: Augment) Error!void {
        const rows = labels_out.len;
        const pixels = batch.len / rows;
        const args: ga.GatherArgs = .{ .start = toU32(start), .rows = toU32(rows), .pixels = toU32(pixels), .max_shift = augment.max_shift, .seed = augment.seed };
        try self.launchFlat(.gather, rows * pixels, .{ address(images), address(labels_src), address(order), address(batch), address(labels_out), args });
    }
};

const testing = std.testing;

fn initForTest() !*Gpu {
    return Gpu.init(testing.allocator, 16 << 20) catch |err| switch (err) {
        error.DriverNotFound => error.SkipZigTest,
        else => err,
    };
}

test "device memory round trip, fill, and the step arena" {
    const gpu = try initForTest();
    defer gpu.deinit();

    const a = try gpu.alloc(f32, 1000);
    var host: [1000]f32 = undefined;
    for (&host, 0..) |*x, i| x.* = @floatFromInt(i);
    try gpu.upload(a, &host);
    var back: [1000]f32 = undefined;
    try gpu.download(&back, a);
    try testing.expectEqualSlices(f32, &host, &back);

    try gpu.fill(a, 2.5);
    try gpu.download(&back, a);
    for (back) |x| try testing.expectEqual(@as(f32, 2.5), x);
    try testing.expectEqual(@as(f32, 2.5), try gpu.scalar(a));

    const s1 = try gpu.scratch(f32, 10);
    const s2 = try gpu.scratch(f32, 10);
    try testing.expect(@intFromPtr(s2.ptr) >= @intFromPtr(s1.ptr) + 40);
    try testing.expectEqual(@as(usize, 0), @intFromPtr(s2.ptr) % 256);
    try gpu.resetStep();
    const s3 = try gpu.scratch(f32, 10);
    try testing.expectEqual(@intFromPtr(s1.ptr), @intFromPtr(s3.ptr));
    try testing.expectError(error.OutOfMemory, gpu.scratch(u8, 1 << 30));
}
