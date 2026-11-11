const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const cuda = @import("cuda.zig");
const conv = @import("math/conv.zig");
const ga = @import("args.zig");
const norm = @import("math/norm.zig");

const ptx = @embedFile("kernels_ptx");

pub const Error = Allocator.Error || error{ NoDevice, DriverNotFound, MissingSymbol, CudaFailed, CudnnNotFound, CudnnFailed, NoAlgorithm };

pub const BinaryOp = enum { add, mul };

pub const Augment = @import("math/augment.zig").Config;
pub const Activation = @import("math/activation.zig").Kind;

pub const MatmulOpts = struct { trans_a: bool = false, trans_b: bool = false, accumulate: bool = false };

pub const ConvGeometry = ga.ConvGeometry;

pub const cudnn = @import("cudnn.zig");

pub const MatmulShape = struct { m: usize, n: usize, k: usize };

pub const BroadcastPlan = struct { dims: [ga.max_rank]u32, a_strides: [ga.max_rank]u32, b_strides: [ga.max_rank]u32 };

pub const ReducePlan = struct { dims: [ga.max_rank]u32, strides: [ga.max_rank]u32, mask: u32 };

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
    packHalf,
    packHalfTransposed,
    im2colHalf,
    sgemmTensor,
    weightsToKrsc,
    convTf32Forward,
    convTf32WeightGrad,
    convTf32WeightGradShort,
    rotateWeightsF32,
    transposeF32,
    addKrscGradient,
    ema,
    swapWeights,
    convGemmForward,
    convGemmWeightGrad,
    convGemmWeightGradShort,
    convGemmForwardWide,
    convGemmWeightGradWide,
    packWeightsRotated,
    elementwise,
    relu,
    reluBackward,
    transposeHalf,
    layerNormRows,
    layerNormBackward,
    layerNormParameters,
    softmaxRows,
    softmaxBackward,
    toHalf,
    fromHalf,
    addFromHalf,
    addToHalf,
    activation,
    activationBackward,
    addRelu,
    addReluBackward,
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
    col2imHalf,
    groupedForward,
    groupedInputGrad,
    groupedWeightGrad,
    maxPool,
    maxPoolBackward,
    avgPool,
    avgPoolBackward,
    maxPoolHalf,
    maxPoolBackwardHalf,
    avgPoolHalf,
    avgPoolBackwardHalf,
    addReluHalf,
    addReluBackwardHalf,
    addReluQuad,
    addReluBackwardQuad,
    batchNormApplyQuad,
    batchNormGradInputQuad,
    dropout,
    columnReduce,
    columnReduceHalf,
    batchNormApplyHalf,
    batchNormGradInputHalf,
    columnCombine,
    splitCombine,
    batchNormFinish,
    batchNormFinishHalf,
    batchNormApply,
    batchNormGradFinish,
    batchNormGradInput,
};
const kernel_count = @typeInfo(Kernel).@"enum".field_names.len;

fn toU32(x: usize) u32 {
    return @intCast(x);
}

fn KernelArg(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => u64,
        .bool => u32,
        .@"enum" => u32,
        .int, .comptime_int => if (T == u64 or T == u32) T else u32,
        else => T,
    };
}

fn kernelArg(value: anytype) KernelArg(@TypeOf(value)) {
    return switch (@typeInfo(@TypeOf(value))) {
        .pointer => @intFromPtr(value.ptr),
        .bool => @intFromBool(value),
        .@"enum" => @backingInt(value),
        .int, .comptime_int => if (KernelArg(@TypeOf(value)) == u32) toU32(value) else value,
        else => value,
    };
}

fn pack(comptime T: type, values: anytype) T {
    var out: T = undefined;
    inline for (@typeInfo(T).@"struct".field_names) |name| {
        const FieldType = @FieldType(T, name);
        const value = @field(values, name);
        @field(out, name) = if (@typeInfo(FieldType) == .int) kernelArg(value) else value;
    }
    return out;
}
fn ceilDiv(x: usize, y: usize) u32 {
    return @intCast((x + y - 1) / y);
}

const slot_fill = kernel_count;
const slot_upload = kernel_count + 1;
const slot_download = kernel_count + 2;
const slot_cudnn = kernel_count + 3;
const slot_count = kernel_count + 4;

fn slotName(slot: usize) []const u8 {
    return switch (slot) {
        slot_fill => "fill (memset)",
        slot_upload => "upload",
        slot_download => "download",
        slot_cudnn => "cudnn convolution",
        else => @tagName(@as(Kernel, @fromBackingInt(@intCast(slot)))),
    };
}

const Profile = struct { io: Io, ns: [slot_count]i96 = @splat(0), calls: [slot_count]u32 = @splat(0) };

pub const Gpu = struct {
    profile: ?Profile = null,
    tensor_cores: bool = false,
    tf32: bool = false,
    tensor_core_min_work: usize = 1 << 22,
    deterministic: bool = false,
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
    cudnn_lib: ?cudnn.Cudnn,
    cudnn_plans: std.ArrayList(PlanEntry),
    cudnn_workspace: usize,
    cudnn_workspace_bytes: usize,

    const PlanEntry = struct { geometry: cudnn.Geometry, plan: cudnn.Plan };

    pub fn init(gpa: Allocator, step_bytes: usize) Error!*Gpu {
        const self = try gpa.create(Gpu);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.profile = null;
        self.tensor_cores = false;
        self.tf32 = false;
        self.tensor_core_min_work = 1 << 22;
        self.deterministic = false;
        self.owned = .empty;
        self.cudnn_lib = null;
        self.cudnn_plans = .empty;
        self.cudnn_workspace = 0;
        self.cudnn_workspace_bytes = 0;
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

    pub fn forTest(step_bytes: usize) !*Gpu {
        return init(std.testing.allocator, step_bytes) catch |err| switch (err) {
            error.DriverNotFound => error.SkipZigTest,
            else => err,
        };
    }

    pub fn deinit(self: *Gpu) void {
        if (self.cudnn_lib) |*lib| {
            for (self.cudnn_plans.items) |entry| lib.destroyPlan(entry.plan);
            lib.close();
        }
        self.cudnn_plans.deinit(self.gpa);
        if (self.cudnn_workspace != 0) self.cu.free(self.cudnn_workspace);
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

    pub fn enableCudnn(self: *Gpu, dir: [:0]const u8) Error!void {
        if (self.cudnn_lib != null) return;
        self.cudnn_lib = try cudnn.Cudnn.open(dir);
    }

    fn cudnnPlan(self: *Gpu, geometry: cudnn.Geometry) Error!cudnn.Plan {
        for (self.cudnn_plans.items) |entry| {
            if (std.meta.eql(entry.geometry, geometry)) return entry.plan;
        }
        const plan = try self.cudnn_lib.?.plan(geometry, .float, 512 << 20, self.deterministic);
        try self.cudnn_plans.append(self.gpa, .{ .geometry = geometry, .plan = plan });
        if (plan.workspaceBytes() > self.cudnn_workspace_bytes) {
            if (self.cudnn_workspace != 0) self.cu.free(self.cudnn_workspace);
            self.cudnn_workspace_bytes = plan.workspaceBytes();
            self.cudnn_workspace = try self.cu.alloc(self.cudnn_workspace_bytes);
        }
        return plan;
    }

    pub fn weightsToKrsc(self: *Gpu, w: []const f32, r: usize, c: usize, k: usize) Error![]f32 {
        const dst = try self.scratch(f32, k * r * r * c);
        try self.launchFlat(.weightsToKrsc, dst.len, .{ w, dst, pack(ga.LayoutArgs, .{ .r = r, .c = c, .k = k }) });
        return dst;
    }

    pub fn addKrscGradient(self: *Gpu, src: []const f32, dst: []f32, r: usize, c: usize, k: usize) Error!void {
        try self.launchFlat(.addKrscGradient, src.len, .{ src, dst, pack(ga.LayoutArgs, .{ .r = r, .c = c, .k = k }) });
    }

    pub fn cudnnForward(self: *Gpu, geometry: cudnn.Geometry, x: []const f32, w_krsc: []const f32, y: []f32) Error!void {
        const plan = try self.cudnnPlan(geometry);
        const started = try self.timerStart();
        try self.cudnn_lib.?.forward(plan, @intFromPtr(x.ptr), @intFromPtr(w_krsc.ptr), @intFromPtr(y.ptr), self.cudnn_workspace, false);
        try self.timerStop(slot_cudnn, started);
    }

    pub fn cudnnBackwardData(self: *Gpu, geometry: cudnn.Geometry, w_krsc: []const f32, dy: []const f32, dx: []f32) Error!void {
        const plan = try self.cudnnPlan(geometry);
        const started = try self.timerStart();
        try self.cudnn_lib.?.backwardData(plan, @intFromPtr(w_krsc.ptr), @intFromPtr(dy.ptr), @intFromPtr(dx.ptr), self.cudnn_workspace, true);
        try self.timerStop(slot_cudnn, started);
    }

    pub fn cudnnBackwardFilter(self: *Gpu, geometry: cudnn.Geometry, x: []const f32, dy: []const f32, dw_krsc: []f32) Error!void {
        const plan = try self.cudnnPlan(geometry);
        const started = try self.timerStart();
        try self.cudnn_lib.?.backwardFilter(plan, @intFromPtr(x.ptr), @intFromPtr(dy.ptr), @intFromPtr(dw_krsc.ptr), self.cudnn_workspace, false);
        try self.timerStop(slot_cudnn, started);
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
        const types = comptime blk: {
            var t: [args.len]type = undefined;
            for (&t, 0..) |*e, i| e.* = KernelArg(@TypeOf(args[i]));
            break :blk t;
        };
        var converted: @Tuple(&types) = undefined;
        var pointers: [args.len]?*anyopaque = undefined;
        inline for (0..args.len) |i| {
            converted[i] = kernelArg(args[i]);
            pointers[i] = &converted[i];
        }
        const started = try self.timerStart();
        try self.cu.launch(self.fns[@backingInt(kernel)], grid, .{ 256, 1, 1 }, &pointers);
        try self.timerStop(@backingInt(kernel), started);
    }

    fn splitCount(self: *const Gpu, wanted: usize) usize {
        return if (self.deterministic) 1 else wanted;
    }

    fn launchFlat(self: *Gpu, kernel: Kernel, items: usize, args: anytype) Error!void {
        try self.launch(kernel, .{ ceilDiv(items, 256), 1, 1 }, args);
    }

    pub fn matmul(self: *Gpu, a: []const f32, b: []const f32, out: []f32, shape: MatmulShape, opts: MatmulOpts) Error!void {
        std.debug.assert(!(opts.trans_a and opts.trans_b));
        if (self.tensorWorthwhile(shape.m, shape.n, shape.k)) return self.matmulTensor(a, b, out, shape, opts);
        const tile: usize = if (shape.n <= 32 and shape.m <= 64) 2 else if (shape.n <= 32) 1 else 0;
        const transpose: usize = if (opts.trans_a) 2 else if (opts.trans_b) 1 else 0;
        const kernel: Kernel = @fromBackingInt(@backingInt(Kernel.sgemmNN) + @as(u5, @intCast(tile * 3 + transpose)));
        const tile_m: usize = if (tile == 2) 32 else 128;
        const tile_n: usize = if (tile == 0) 128 else 32;
        const blocks = @as(usize, ceilDiv(shape.n, tile_n)) * ceilDiv(shape.m, tile_m);
        const splits = self.splitCount(@max(1, @min(shape.k / 256, (128 + blocks - 1) / blocks)));
        const chunk = std.mem.alignForward(usize, (shape.k + splits - 1) / splits, 8);
        if (splits > 1 and !opts.accumulate) try self.fill(out, 0);
        const kernel_shape = pack(ga.GemmShape, .{ .m = shape.m, .n = shape.n, .k = shape.k, .accumulate = opts.accumulate, .chunk = chunk });
        try self.launch(kernel, .{ ceilDiv(shape.n, tile_n), ceilDiv(shape.m, tile_m), ceilDiv(shape.k, chunk) }, .{ a, b, out, kernel_shape });
    }

    pub const Packed = struct { data: []u16, rows_padded: usize, cols_padded: usize };

    pub fn tensorWorthwhile(self: *Gpu, m: usize, n: usize, k: usize) bool {
        return self.tensor_cores and m * n * k >= self.tensor_core_min_work;
    }

    pub fn packMatrix(self: *Gpu, src: anytype, rows: usize, cols: usize, rows_padded: usize, cols_padded: usize, transpose: bool) Error!Packed {
        const dst = try self.scratch(u16, rows_padded * cols_padded);
        const args = pack(ga.PackArgs, .{ .rows = rows, .cols = cols, .rows_padded = rows_padded, .cols_padded = cols_padded, .src_stride = if (transpose) rows else cols, .transpose = transpose });
        if (transpose) {
            try self.launch(if (@typeInfo(@TypeOf(src)).pointer.child == u16) .transposeHalf else .packHalfTransposed, .{ toU32(cols_padded / 32), toU32(rows_padded / 32), 1 }, .{ src, dst, args });
        } else {
            try self.launchFlat(.packHalf, rows_padded * cols_padded, .{ src, dst, args });
        }
        return .{ .data = dst, .rows_padded = rows_padded, .cols_padded = cols_padded };
    }

    pub fn im2colPacked(self: *Gpu, x: []const f32, d: conv.Dims) Error!Packed {
        const rows = conv.workItems(.im2col, d) / (d.k * d.k * d.c);
        const rows_padded = std.mem.alignForward(usize, rows, 128);
        const cols_padded = std.mem.alignForward(usize, d.k * d.k * d.c, 64);
        const dst = try self.scratch(u16, rows_padded * cols_padded);
        try self.launch(.im2colHalf, .{ toU32(rows_padded), 1, 1 }, .{ x, dst, d, cols_padded });
        return .{ .data = dst, .rows_padded = rows_padded, .cols_padded = cols_padded };
    }

    pub fn tf32Worthwhile(self: *Gpu, m: usize, n: usize, k: usize) bool {
        return self.tf32 and m * n * k >= self.tensor_core_min_work;
    }

    pub fn rotateWeightsF32(self: *Gpu, w: []const f32, k: usize, c_in: usize, c_out: usize) Error![]f32 {
        const rows = k * k * c_out;
        const cols_padded = std.mem.alignForward(usize, c_in, 64);
        const dst = try self.scratch(f32, rows * cols_padded);
        try self.launchFlat(.rotateWeightsF32, dst.len, .{ w, dst, pack(ga.RotateArgs, .{ .k = k, .c_in = c_in, .c_out = c_out, .rows_padded = rows, .cols_padded = cols_padded }) });
        return dst;
    }

    pub fn transposeF32(self: *Gpu, src: []const f32, rows: usize, cols: usize, rows_padded: usize, cols_padded: usize) Error![]f32 {
        const dst = try self.scratch(f32, rows_padded * cols_padded);
        const args = pack(ga.PackArgs, .{ .rows = rows, .cols = cols, .rows_padded = rows_padded, .cols_padded = cols_padded, .src_stride = rows, .transpose = true });
        try self.launch(.transposeF32, .{ toU32(cols_padded / 32), toU32(rows_padded / 32), 1 }, .{ src, dst, args });
        return dst;
    }

    pub fn convTf32Forward(self: *Gpu, x: []const f32, w: []const f32, w_pitch: usize, out: []f32, geo: ga.ConvGeometry, accumulate: bool) Error!void {
        const c_out: usize = geo.c_out;
        const m = @as(usize, geo.n) * geo.out_h * geo.out_w;
        const k_total = @as(usize, geo.k) * geo.k * geo.c;
        const kernel_shape = pack(ga.TensorShape, .{ .m = m, .n = c_out, .k_padded = k_total, .n_padded = w_pitch, .accumulate = accumulate, .chunk = k_total, .transpose_out = false, .output_half = false, .split_stride = 0 });
        try self.launch(.convTf32Forward, .{ ceilDiv(c_out, 64), ceilDiv(m, 128), 1 }, .{ x, w, out, kernel_shape, geo });
    }

    pub fn convTf32WeightGrad(self: *Gpu, dy_t: []const f32, x: []const f32, out: []f32, geo: ga.ConvGeometry) Error!void {
        const c_out: usize = geo.c_out;
        const taps = @as(usize, geo.k) * geo.k * geo.c;
        const positions_padded = std.mem.alignForward(usize, geo.positions, 32);
        const short = c_out <= 64;
        const tile_rows: usize = if (short) 64 else 128;
        const tile_cols: usize = if (short) 128 else 64;
        const blocks = @as(usize, ceilDiv(c_out, tile_rows)) * ceilDiv(taps, tile_cols);
        const splits = @max(1, @min(positions_padded / 256, (128 + blocks - 1) / blocks));
        const chunk = std.mem.alignForward(usize, (positions_padded + splits - 1) / splits, 16);
        const kernel_shape = pack(ga.TensorShape, .{ .m = c_out, .n = taps, .k_padded = positions_padded, .n_padded = taps, .accumulate = true, .chunk = chunk, .transpose_out = true, .output_half = false, .split_stride = 0 });
        const kernel: Kernel = if (short) .convTf32WeightGradShort else .convTf32WeightGrad;
        const mark = self.step_used;
        defer self.step_used = mark;
        const parts = ceilDiv(positions_padded, chunk);
        const target, const shape = try self.orderedTarget(out, kernel_shape, parts);
        try self.launch(kernel, .{ ceilDiv(taps, tile_cols), ceilDiv(c_out, tile_rows), parts }, .{ dy_t, x, target, shape, geo });
        try self.combineSplits(target, out, shape, parts, true);
    }

    pub fn packRotatedWeights(self: *Gpu, w: []const f32, k: usize, c_in: usize, c_out: usize) Error!Packed {
        const rows = k * k * c_out;
        const cols_padded = std.mem.alignForward(usize, c_in, 64);
        const dst = try self.scratch(u16, rows * cols_padded);
        try self.launchFlat(.packWeightsRotated, dst.len, .{ w, dst, pack(ga.RotateArgs, .{ .k = k, .c_in = c_in, .c_out = c_out, .rows_padded = rows, .cols_padded = cols_padded }) });
        return .{ .data = dst, .rows_padded = rows, .cols_padded = cols_padded };
    }

    pub fn convImplicitForward(self: *Gpu, x_half: []const u16, w: Packed, out: anytype, geo: ga.ConvGeometry, accumulate: bool) Error!void {
        const half_out = @typeInfo(@TypeOf(out)).pointer.child == u16;
        const c_out: usize = geo.c_out;
        const m = @as(usize, geo.n) * geo.out_h * geo.out_w;
        const kernel_shape = pack(ga.TensorShape, .{ .m = m, .n = c_out, .k_padded = w.rows_padded, .n_padded = w.cols_padded, .accumulate = accumulate, .chunk = w.rows_padded, .transpose_out = false, .output_half = half_out, .split_stride = 0 });
        const wide = c_out % 128 == 0 and ceilDiv(m, 128) * (c_out / 128) >= 96;
        try self.launch(if (wide) .convGemmForwardWide else .convGemmForward, .{ ceilDiv(c_out, if (wide) 128 else 64), ceilDiv(m, 128), 1 }, .{ x_half, w.data, out, kernel_shape, geo });
    }

    pub fn convImplicitWeightGrad(self: *Gpu, dy_t: Packed, x_half: []const u16, out: []f32, geo: ga.ConvGeometry) Error!void {
        const c_out: usize = geo.c_out;
        const taps = @as(usize, geo.k) * geo.k * geo.c;
        const short = c_out <= 64;
        const wide = c_out % 128 == 0;
        const tile_rows: usize = if (short) 64 else 128;
        const tile_cols: usize = if (short or wide) 128 else 64;
        const n_padded = std.mem.alignForward(usize, taps, tile_cols);
        const blocks = ceilDiv(c_out, tile_rows) * (n_padded / tile_cols);
        const splits = @max(1, @min(dy_t.cols_padded / 256, (128 + blocks - 1) / blocks));
        const chunk = std.mem.alignForward(usize, (dy_t.cols_padded + splits - 1) / splits, 32);
        const kernel_shape = pack(ga.TensorShape, .{ .m = c_out, .n = taps, .k_padded = dy_t.cols_padded, .n_padded = n_padded, .accumulate = true, .chunk = chunk, .transpose_out = true, .output_half = false, .split_stride = 0 });
        const kernel: Kernel = if (short) .convGemmWeightGradShort else if (wide) .convGemmWeightGradWide else .convGemmWeightGrad;
        const mark = self.step_used;
        defer self.step_used = mark;
        const parts = ceilDiv(dy_t.cols_padded, chunk);
        const target, const shape = try self.orderedTarget(out, kernel_shape, parts);
        try self.launch(kernel, .{ toU32(n_padded / tile_cols), ceilDiv(c_out, tile_rows), parts }, .{ dy_t.data, x_half, target, shape, geo });
        try self.combineSplits(target, out, shape, parts, true);
    }

    fn orderedTarget(self: *Gpu, out: anytype, shape: ga.TensorShape, parts: u32) Error!struct { @TypeOf(out), ga.TensorShape } {
        if (!self.deterministic or parts == 1) return .{ out, shape };
        var ordered = shape;
        ordered.split_stride = @intCast(out.len);
        return .{ try self.scratch(f32, @as(usize, parts) * out.len), ordered };
    }

    fn combineSplits(self: *Gpu, workspace: []f32, out: []f32, shape: ga.TensorShape, parts: u32, accumulate: bool) Error!void {
        if (shape.split_stride == 0) return;
        try self.launchFlat(.splitCombine, out.len, .{ workspace, out, parts, shape.split_stride, accumulate });
    }

    pub fn matmulPacked(self: *Gpu, a: Packed, b: Packed, out: anytype, shape: MatmulShape, accumulate: bool, transpose_out: bool) Error!void {
        const half_out = @typeInfo(@TypeOf(out)).pointer.child == u16;
        std.debug.assert(a.cols_padded == b.rows_padded and a.rows_padded % 128 == 0 and b.cols_padded % 64 == 0 and a.cols_padded % 32 == 0);
        const blocks = (a.rows_padded / 128) * (b.cols_padded / 64);
        const splits = @max(1, @min(a.cols_padded / 256, (128 + blocks - 1) / blocks));
        const chunk = std.mem.alignForward(usize, (a.cols_padded + splits - 1) / splits, 32);
        std.debug.assert(!half_out or (chunk >= a.cols_padded and !accumulate and !transpose_out));
        if (!half_out and chunk < a.cols_padded and !accumulate) try self.fill(out, 0);
        const kernel_shape = pack(ga.TensorShape, .{ .m = shape.m, .n = shape.n, .k_padded = a.cols_padded, .n_padded = b.cols_padded, .accumulate = accumulate, .chunk = chunk, .transpose_out = transpose_out, .output_half = half_out, .split_stride = 0 });
        const mark = self.step_used;
        defer self.step_used = mark;
        const parts = ceilDiv(a.cols_padded, chunk);
        const target, const launched = if (half_out) .{ out, kernel_shape } else try self.orderedTarget(out, kernel_shape, parts);
        try self.launch(.sgemmTensor, .{ toU32(b.cols_padded / 64), toU32(a.rows_padded / 128), parts }, .{ a.data, b.data, target, launched });
        if (!half_out) try self.combineSplits(target, out, launched, parts, accumulate);
    }

    fn matmulTensor(self: *Gpu, a: []const f32, b: []const f32, out: []f32, shape: MatmulShape, opts: MatmulOpts) Error!void {
        const mark = self.step_used;
        defer self.step_used = mark;
        const k_padded = std.mem.alignForward(usize, shape.k, 32);
        const a_packed = try self.packMatrix(a, shape.m, shape.k, std.mem.alignForward(usize, shape.m, 128), k_padded, opts.trans_a);
        const b_packed = try self.packMatrix(b, shape.k, shape.n, k_padded, std.mem.alignForward(usize, shape.n, 64), opts.trans_b);
        try self.matmulPacked(a_packed, b_packed, out, shape, opts.accumulate, false);
    }
    pub fn elementwise(self: *Gpu, op: BinaryOp, a: []const f32, b: []const f32, out: []f32, plan: BroadcastPlan) Error!void {
        const args = pack(ga.ElementwiseArgs, .{ .len = out.len, .multiply = op == .mul, .dims = plan.dims, .a_strides = plan.a_strides, .b_strides = plan.b_strides });
        try self.launchFlat(.elementwise, out.len, .{ a, b, out, args });
    }

    pub fn relu(self: *Gpu, x: []const f32, out: []f32) Error!void {
        try self.launchFlat(.relu, x.len, .{ x, out, x.len });
    }

    pub fn reluBackward(self: *Gpu, x: []const f32, g_out: []const f32, g_in: []f32) Error!void {
        try self.launchFlat(.reluBackward, x.len, .{ x, g_out, g_in, x.len });
    }

    pub const LayerNorm = struct { x: []const f32, gamma: []const f32, mean: []f32, inv_std: []f32 };

    pub fn layerNorm(self: *Gpu, p: LayerNorm, beta: []const f32, y: []f32) Error!void {
        try self.launchFlat(.layerNormRows, p.mean.len, .{ p.x, y, p.gamma, beta, p.mean, p.inv_std, p.mean.len, p.gamma.len });
    }

    pub fn layerNormBackward(self: *Gpu, p: LayerNorm, dy: []const f32, dx: ?[]f32, dgamma: ?[]f32, dbeta: ?[]f32) Error!void {
        const rows = p.mean.len;
        const cols = p.gamma.len;
        if (dx) |out| try self.launchFlat(.layerNormBackward, rows, .{ p.x, dy, out, p.gamma, p.mean, p.inv_std, rows, cols });
        if (dgamma) |g| try self.launchFlat(.layerNormParameters, cols, .{ p.x, dy, p.mean, p.inv_std, g, dbeta.?, rows, cols });
    }

    pub fn softmax(self: *Gpu, x: []const f32, y: []f32, cols: usize, period: usize) Error!void {
        try self.launchFlat(.softmaxRows, x.len / cols, .{ x, y, x.len / cols, cols, period });
    }

    pub fn softmaxBackward(self: *Gpu, y: []const f32, dy: []const f32, dx: []f32, cols: usize) Error!void {
        try self.launchFlat(.softmaxBackward, y.len / cols, .{ y, dy, dx, y.len / cols, cols });
    }

    pub fn toHalf(self: *Gpu, x: []const f32, y: []u16) Error!void {
        try self.launchFlat(.toHalf, x.len, .{ x, y, x.len });
    }

    pub fn fromHalf(self: *Gpu, x: []const u16, y: []f32) Error!void {
        try self.launchFlat(.fromHalf, x.len, .{ x, y, x.len });
    }

    pub fn addFromHalf(self: *Gpu, x: []const u16, y: []f32) Error!void {
        try self.launchFlat(.addFromHalf, x.len, .{ x, y, x.len });
    }

    pub fn addToHalf(self: *Gpu, x: []const f32, y: []u16) Error!void {
        try self.launchFlat(.addToHalf, x.len, .{ x, y, x.len });
    }

    pub fn activation(self: *Gpu, x: []const f32, out: []f32, kind: Activation) Error!void {
        try self.launchFlat(.activation, x.len, .{ x, out, x.len, kind });
    }

    pub fn activationBackward(self: *Gpu, x: []const f32, y: []const f32, g_out: []const f32, g_in: []f32, kind: Activation) Error!void {
        try self.launchFlat(.activationBackward, x.len, .{ x, y, g_out, g_in, x.len, kind });
    }

    pub fn addRelu(self: *Gpu, a: anytype, b: anytype, out: anytype) Error!void {
        const E = @typeInfo(@TypeOf(out)).pointer.child;
        if (E == u16 and out.len % 4 == 0) return self.launchFlat(.addReluQuad, out.len / 4, .{ a, b, out, out.len });
        try self.launchFlat(if (E == u16) .addReluHalf else .addRelu, out.len, .{ a, b, out, out.len });
    }

    pub fn addReluBackward(self: *Gpu, out: anytype, g: anytype, grad_a: anytype, grad_b: anytype, a_needs_grad: bool, b_needs_grad: bool) Error!void {
        const E = @typeInfo(@TypeOf(out)).pointer.child;
        const flags: u32 = @as(u32, @intFromBool(a_needs_grad)) | @as(u32, @intFromBool(b_needs_grad)) << 1;
        if (E == u16 and out.len % 4 == 0) return self.launchFlat(.addReluBackwardQuad, out.len / 4, .{ out, g, grad_a, grad_b, out.len, flags });
        try self.launchFlat(if (E == u16) .addReluBackwardHalf else .addReluBackward, out.len, .{ out, g, grad_a, grad_b, out.len, flags });
    }

    pub fn addInPlace(self: *Gpu, dst: []f32, src: []const f32) Error!void {
        try self.launchFlat(.addInPlace, dst.len, .{ dst, src, dst.len });
    }

    pub fn sumTo(self: *Gpu, in: []const f32, out: []f32, plan: ReducePlan, accumulate: bool) Error!void {
        const last = ga.max_rank - 1;
        if (plan.mask == (1 << last) - 1 and plan.strides[last] == 1 and plan.dims[last] == out.len) {
            if (!accumulate) try self.fill(out, 0);
            return self.columnSums(in, out);
        }
        const args = pack(ga.SumArgs, .{ .out_len = out.len, .reduce_count = in.len / out.len, .mask = plan.mask, .accumulate = accumulate, .dims = plan.dims, .strides = plan.strides });
        try self.launch(.sumTo, .{ toU32(out.len), 1, 1 }, .{ in, out, args });
    }

    pub fn crossEntropy(self: *Gpu, logits: []const f32, logp: []f32, labels: []const u8, loss: []f32, cols: usize, smoothing: f32) Error!void {
        const rows = labels.len;
        const row_loss = try self.scratch(f32, rows);
        try self.launchFlat(.crossEntropyRows, rows, .{ logits, logp, labels, row_loss, rows, cols, smoothing });
        try self.launchFlat(.crossEntropyMean, 1, .{ row_loss, loss, rows });
    }

    pub fn crossEntropyBackward(self: *Gpu, logp: []const f32, labels: []const u8, grad: []f32, loss_grad: []const f32, cols: usize, smoothing: f32) Error!void {
        const rows = labels.len;
        try self.launchFlat(.crossEntropyBackward, rows * cols, .{ logp, labels, grad, loss_grad, rows, cols, smoothing });
    }

    pub fn convKernel(self: *Gpu, comptime op: conv.Op, slices: anytype, d: conv.Dims) Error!void {
        const len = conv.workItems(op, d);
        try self.launchFlat(@field(Kernel, @tagName(op)), len, slices ++ .{ len, d });
    }

    const ColumnKind = enum(u32) { sum, gradient_sums, shifted_moments, masked_gradient_sums };

    fn columnReduce(self: *Gpu, comptime E: type, args: ga.ColumnArgs) Error!void {
        const blocks: u32 = @min(256, ceilDiv(args.rows, 8 * 16));
        var launched = args;
        launched.row_stride = blocks * 8;
        launched.ordered = @intFromBool(self.deterministic);
        const mark = self.step_used;
        defer self.step_used = mark;
        const partial: []f32 = if (self.deterministic) try self.scratch(f32, @as(usize, blocks) * 2 * args.channels) else &.{};
        launched.partial = partial.ptr;
        try self.launch(if (E == u16) .columnReduceHalf else .columnReduce, .{ blocks, ceilDiv(args.channels, 32), 1 }, .{launched});
        if (self.deterministic) {
            const width = args.channels * @as(usize, if (args.kind == 0) 1 else 2);
            try self.launchFlat(.columnCombine, width, .{ partial, args.out[0..width], blocks, width, 2 * args.channels });
        }
    }

    fn columns(a: anytype, out: []f32, kind: ColumnKind) ga.ColumnArgs {
        const channels = if (kind == .sum) out.len else out.len / 2;
        return .{ .a = @ptrCast(@alignCast(a.ptr)), .b = @ptrCast(@alignCast(a.ptr)), .mean = out.ptr, .inv_std = out.ptr, .mask = @ptrCast(@alignCast(a.ptr)), .out = out.ptr, .partial = out.ptr, .ordered = 0, .rows = @intCast(a.len / channels), .channels = @intCast(channels), .row_stride = 0, .kind = @backingInt(kind), .scale = 1 };
    }

    pub fn columnSums(self: *Gpu, in: []const f32, out: []f32) Error!void {
        try self.columnReduce(f32, columns(in, out, .sum));
    }

    pub fn batchNormStats(self: *Gpu, x: anytype, st: norm.Stats, c: usize, training: bool) Error!void {
        const E = @typeInfo(@TypeOf(x)).pointer.child;
        const sums = try self.scratchZeroed(f32, 2 * c);
        if (training) try self.columnReduce(E, columns(x, sums, .shifted_moments));
        try self.launchFlat(if (E == u16) .batchNormFinishHalf else .batchNormFinish, c, .{ sums, x[0..c], st, norm.Layout{ .len = @intCast(x.len), .channels = @intCast(c), .flag = @intFromBool(training) } });
    }

    pub fn batchNormApply(self: *Gpu, x: anytype, y: anytype, p: norm.Affine, l: norm.Layout) Error!void {
        const E = @typeInfo(@TypeOf(x)).pointer.child;
        if (E == u16 and l.channels % 4 == 0 and x.len % 4 == 0) return self.launchFlat(.batchNormApplyQuad, x.len / 4, .{ x, y, p, l });
        try self.launchFlat(if (E == u16) .batchNormApplyHalf else .batchNormApply, x.len, .{ x, y, p, l });
    }

    pub fn batchNormGradSums(self: *Gpu, x: anytype, dy: anytype, p: norm.Affine, mask: anytype, out: norm.GradOut) Error!void {
        const E = @typeInfo(@TypeOf(x)).pointer.child;
        var args = columns(dy, out.sums, if (mask.len != 0) .masked_gradient_sums else .gradient_sums);
        args.b = @ptrCast(@alignCast(x.ptr));
        args.mean = p.mean;
        args.inv_std = p.inv_std;
        if (mask.len != 0) args.mask = @ptrCast(@alignCast(mask.ptr));
        try self.columnReduce(E, args);
        try self.launchFlat(.batchNormGradFinish, out.beta.len, .{ out.sums, out.gamma, out.beta, out.beta.len });
    }

    pub fn batchNormGradInput(self: *Gpu, io: anytype, p: norm.Affine, l: norm.Layout) Error!void {
        const E = @typeInfo(@TypeOf(io.x)).pointer.child;
        if (E == u16 and l.channels % 4 == 0 and l.len % 4 == 0) return self.launchFlat(.batchNormGradInputQuad, l.len / 4, .{ io, p, l });
        try self.launchFlat(if (E == u16) .batchNormGradInputHalf else .batchNormGradInput, l.len, .{ io, p, l });
    }
    pub fn dropout(self: *Gpu, src: []const f32, dst: []f32, seed: u64, keep: f32, accumulate: bool) Error!void {
        try self.launchFlat(.dropout, src.len, .{ src, dst, src.len, seed, keep, accumulate });
    }

    fn launchTable(self: *Gpu, kernel: Kernel, table: ga.TensorTable, args: anytype) Error!void {
        var blocks: u32 = 0;
        var launched = table;
        for (table.len[0..table.count], launched.first_block[0..table.count]) |len, *first| {
            first.* = blocks;
            blocks += ceilDiv(len, 1024);
        }
        try self.launch(kernel, .{ blocks, 1, 1 }, .{launched} ++ args);
    }

    pub fn zeroGrads(self: *Gpu, table: ga.TensorTable) Error!void {
        try self.launchTable(.zeroGrads, table, .{});
    }

    pub fn sgd(self: *Gpu, table: ga.TensorTable, lr: f32) Error!void {
        try self.launchTable(.sgd, table, .{lr});
    }

    pub fn ema(self: *Gpu, table: ga.TensorTable, decay: f32) Error!void {
        try self.launchTable(.ema, table, .{decay});
    }

    pub fn swapWeights(self: *Gpu, table: ga.TensorTable) Error!void {
        try self.launchTable(.swapWeights, table, .{});
    }

    pub fn adam(self: *Gpu, table: ga.TensorTable, hp: ga.AdamArgs) Error!void {
        try self.launchTable(.adam, table, .{hp});
    }

    pub const GatherFrom = struct { images: []const f32, labels: []const u8, order: []const u32 };
    pub const GatherTo = struct { batch: []f32, labels: []u8, other_labels: []u8 };

    pub fn gather(self: *Gpu, from: GatherFrom, to: GatherTo, start: usize, augment: Augment) Error!void {
        const rows = to.labels.len;
        const pixels = to.batch.len / rows;
        const args = pack(ga.GatherArgs, .{ .start = start, .rows = rows, .pixels = pixels, .max_shift = augment.max_shift, .max_angle = augment.max_angle, .max_scale = augment.max_scale, .channels = augment.channels, .flip = augment.flip, .mirror = augment.mirror, .cutout = augment.cutout, .x0 = augment.box.x0, .y0 = augment.box.y0, .x1 = augment.box.x1, .y1 = augment.box.y1, .seed = augment.seed });
        try self.launchFlat(.gather, rows * pixels, .{ from.images, from.labels, from.order, to.batch, to.labels, to.other_labels, args });
    }
};

const testing = std.testing;

test "device memory round trip, fill, and the step arena" {
    const gpu = try Gpu.forTest(16 << 20);
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

test "deterministic mode gives bit-identical results on every run, for split-K products and batch-norm statistics" {
    const gpu = try Gpu.forTest(16 << 20);
    defer gpu.deinit();
    gpu.tensor_core_min_work = 0;
    gpu.deterministic = true;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(17);
    const rng = prng.random();

    const m = 64;
    const n = 64;
    const k = 16384;
    const a = try gpa.alloc(f32, m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, k * n);
    defer gpa.free(b);
    for (a) |*v| v.* = rng.floatNorm(f32);
    for (b) |*v| v.* = rng.floatNorm(f32);
    const da = try gpu.alloc(f32, a.len);
    const db = try gpu.alloc(f32, b.len);
    const dc = try gpu.alloc(f32, m * n);
    try gpu.upload(da, a);
    try gpu.upload(db, b);

    const rows = 20000;
    const channels = 48;
    const x = try gpa.alloc(f32, rows * channels);
    defer gpa.free(x);
    for (x) |*v| v.* = rng.floatNorm(f32) * 3 + 1;
    const dx = try gpu.alloc(f32, x.len);
    try gpu.upload(dx, x);
    const mean = try gpu.alloc(f32, channels);
    const inv_std = try gpu.alloc(f32, channels);
    const running_mean = try gpu.alloc(f32, channels);
    const running_var = try gpu.alloc(f32, channels);
    try gpu.fill(running_mean, 0);
    try gpu.fill(running_var, 1);

    var first_product: [m * n]f32 = undefined;
    var first_stats: [2 * channels]f32 = undefined;
    for (0..2) |tensor_cores| {
        gpu.tensor_cores = tensor_cores == 1;
        for (0..6) |run| {
            try gpu.matmul(da, db, dc, .{ .m = m, .n = n, .k = k }, .{});
            try gpu.batchNormStats(dx, .{ .mean = mean.ptr, .inv_std = inv_std.ptr, .running_mean = running_mean.ptr, .running_var = running_var.ptr }, channels, true);
            var product: [m * n]f32 = undefined;
            var stats: [2 * channels]f32 = undefined;
            try gpu.download(&product, dc);
            try gpu.download(stats[0..channels], mean);
            try gpu.download(stats[channels..], inv_std);
            if (run == 0) {
                first_product = product;
                first_stats = stats;
            } else {
                try testing.expectEqualSlices(f32, &first_product, &product);
                try testing.expectEqualSlices(f32, &first_stats, &stats);
            }
        }
        const tolerance: f32 = if (tensor_cores == 1) 2.0 else 0.05;
        for (0..4) |i| {
            for (0..n) |j| {
                var want: f64 = 0;
                for (0..k) |p| want += @as(f64, a[i * k + p]) * b[p * n + j];
                try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), first_product[i * n + j], tolerance);
            }
        }
    }
    for (0..channels) |ch| {
        var sum: f64 = 0;
        var square: f64 = 0;
        for (0..rows) |r| sum += x[r * channels + ch];
        const mu = sum / rows;
        for (0..rows) |r| square += (x[r * channels + ch] - mu) * (x[r * channels + ch] - mu);
        try testing.expectApproxEqAbs(@as(f32, @floatCast(mu)), first_stats[ch], 1e-4);
        try testing.expectApproxEqAbs(@as(f32, @floatCast(1 / @sqrt(square / rows + norm.eps))), first_stats[channels + ch], 1e-4);
    }
}
