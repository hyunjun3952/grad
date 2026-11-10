const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda.zig");

const ptx = @embedFile("kernels_ptx");

pub const Error = Allocator.Error || error{ DriverNotFound, MissingSymbol, CudaFailed };

pub const BinaryOp = enum { add, mul };

const Kernel = enum {
    sgemm_nn,
    sgemm_nt,
    sgemm_tn,
    bin_add,
    bin_mul,
    relu,
    relu_bwd,
    add_inplace,
    sum_to,
    ce_rows,
    ce_mean,
    ce_bwd,
    sgd,
    adam,
    gather,
};
const kernel_count = 15;

fn dev(slice: anytype) u64 {
    return @intFromPtr(slice.ptr);
}

fn u32Of(x: usize) u32 {
    return @intCast(x);
}

fn ceilDiv(x: usize, y: usize) u32 {
    return @intCast((x + y - 1) / y);
}

pub const Gpu = struct {
    gpa: Allocator,
    cu: cuda.Cuda,
    fns: [kernel_count]cuda.Handle,
    owned: std.ArrayList(cuda.DevicePtr),
    step_base: usize,
    step_size: usize,
    step_used: usize,

    pub fn init(gpa: Allocator, step_bytes: usize) Error!*Gpu {
        const self = try gpa.create(Gpu);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.owned = .empty;
        self.cu = try cuda.Cuda.open();
        errdefer self.cu.close();

        const module = try self.cu.loadModule(gpa, ptx);
        inline for (0..kernel_count) |i| {
            self.fns[i] = try self.cu.function(module, @tagName(@as(Kernel, @fromBackingInt(@intCast(i)))));
        }
        self.step_base = try self.cu.alloc(step_bytes);
        self.step_size = step_bytes;
        self.step_used = 0;
        return self;
    }

    pub fn deinit(self: *Gpu) void {
        for (self.owned.items) |ptr| self.cu.free(ptr);
        self.cu.free(self.step_base);
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

    pub fn resetStep(self: *Gpu) void {
        self.step_used = 0;
    }

    pub fn upload(self: *Gpu, dst: anytype, src: anytype) Error!void {
        std.debug.assert(dst.len == src.len);
        try self.cu.upload(@intFromPtr(dst.ptr), src);
    }

    pub fn download(self: *Gpu, dst: anytype, src: anytype) Error!void {
        std.debug.assert(dst.len == src.len);
        try self.cu.download(dst, @intFromPtr(src.ptr));
    }

    pub fn fill(self: *Gpu, dst: []f32, value: f32) Error!void {
        try self.cu.memsetD32(@intFromPtr(dst.ptr), @bitCast(value), dst.len);
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
        inline for (0..copy.len) |i| pointers[i] = &copy[i];
        try self.cu.launch(self.fns[@backingInt(kernel)], grid, .{ 256, 1, 1 }, &pointers);
    }

    fn launch1d(self: *Gpu, kernel: Kernel, items: usize, args: anytype) Error!void {
        try self.launch(kernel, .{ ceilDiv(items, 256), 1, 1 }, args);
    }

    pub fn matmul(
        self: *Gpu,
        a: []const f32,
        b: []const f32,
        out: []f32,
        trans_a: bool,
        trans_b: bool,
        accumulate: bool,
        m: usize,
        n: usize,
        k: usize,
    ) Error!void {
        std.debug.assert(!(trans_a and trans_b));
        const kernel: Kernel = if (trans_a) .sgemm_tn else if (trans_b) .sgemm_nt else .sgemm_nn;
        const flag: u32 = @intFromBool(accumulate);
        try self.launch(kernel, .{ ceilDiv(n, 64), ceilDiv(m, 64), 1 }, .{ dev(a), dev(b), dev(out), u32Of(m), u32Of(n), u32Of(k), flag });
    }

    pub fn binary(self: *Gpu, op: BinaryOp, a: []const f32, b: []const f32, out: []f32, dims: [4]u32, sa: [4]u32, sb: [4]u32) Error!void {
        const kernel: Kernel = switch (op) {
            .add => .bin_add,
            .mul => .bin_mul,
        };
        try self.launch1d(kernel, out.len, .{
            dev(a), dev(b), dev(out), u32Of(out.len), dims[1], dims[2], dims[3],
            sa[0],  sa[1],  sa[2],    sa[3],          sb[0],   sb[1],   sb[2],
            sb[3],
        });
    }

    pub fn relu(self: *Gpu, x: []const f32, out: []f32) Error!void {
        try self.launch1d(.relu, x.len, .{ dev(x), dev(out), u32Of(x.len) });
    }

    pub fn reluBackward(self: *Gpu, x: []const f32, g_out: []const f32, g_in: []f32) Error!void {
        try self.launch1d(.relu_bwd, x.len, .{ dev(x), dev(g_out), dev(g_in), u32Of(x.len) });
    }

    pub fn addInPlace(self: *Gpu, dst: []f32, src: []const f32) Error!void {
        try self.launch1d(.add_inplace, dst.len, .{ dev(dst), dev(src), u32Of(dst.len) });
    }

    pub fn sumTo(self: *Gpu, in: []const f32, out: []f32, dims: [4]u32, strides: [4]u32, reduced: u32, accumulate: bool) Error!void {
        const flag: u32 = @intFromBool(accumulate);
        try self.launch1d(.sum_to, out.len, .{
            dev(in),    dev(out),   u32Of(out.len), u32Of(in.len / out.len), reduced,
            flag,       dims[0],    dims[1],        dims[2],                 dims[3],
            strides[0], strides[1], strides[2],     strides[3],
        });
    }

    pub fn crossEntropy(self: *Gpu, logits: []const f32, logp: []f32, labels: []const u8, loss: []f32, rows: usize, cols: usize) Error!void {
        const row_loss = try self.scratch(f32, rows);
        try self.launch1d(.ce_rows, rows, .{ dev(logits), dev(logp), dev(labels), dev(row_loss), u32Of(rows), u32Of(cols) });
        try self.launch1d(.ce_mean, 1, .{ dev(row_loss), dev(loss), u32Of(rows) });
    }

    pub fn crossEntropyBackward(self: *Gpu, logp: []const f32, labels: []const u8, grad: []f32, loss_grad: []const f32, rows: usize, cols: usize) Error!void {
        try self.launch1d(.ce_bwd, rows * cols, .{ dev(logp), dev(labels), dev(grad), dev(loss_grad), u32Of(rows), u32Of(cols) });
    }

    pub fn sgd(self: *Gpu, w: []f32, g: []const f32, lr: f32) Error!void {
        try self.launch1d(.sgd, w.len, .{ dev(w), dev(g), lr, u32Of(w.len) });
    }

    pub fn adam(self: *Gpu, w: []f32, g: []const f32, m: []f32, v: []f32, b1: f32, b2: f32, step: f32, inv_sqrt_c2: f32, eps: f32) Error!void {
        try self.launch1d(.adam, w.len, .{ dev(w), dev(g), dev(m), dev(v), u32Of(w.len), b1, b2, step, inv_sqrt_c2, eps });
    }

    pub fn gather(self: *Gpu, images: []const f32, labels_src: []const u8, order: []const u32, batch: []f32, labels_out: []u8, start: usize, rows: usize, pixels: usize) Error!void {
        try self.launch1d(.gather, rows * pixels, .{
            dev(images),  dev(labels_src), dev(order),    dev(batch), dev(labels_out),
            u32Of(start), u32Of(rows),     u32Of(pixels),
        });
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
    gpu.resetStep();
    const s3 = try gpu.scratch(f32, 10);
    try testing.expectEqual(@intFromPtr(s1.ptr), @intFromPtr(s3.ptr));
    try testing.expectError(error.OutOfMemory, gpu.scratch(u8, 1 << 30));
}
