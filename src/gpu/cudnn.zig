const std = @import("std");
const Allocator = std.mem.Allocator;
const dynlib = @import("dynlib.zig");

const Status = c_int;
const Handle = ?*anyopaque;

pub const Error = error{ CudnnNotFound, MissingSymbol, CudnnFailed, NoAlgorithm } || Allocator.Error;

pub const DataType = enum(c_int) { float = 0, half = 2, bfloat16 = 9 };

const Perf = extern struct {
    algo: c_int,
    status: Status,
    time: f32,
    memory: usize,
    determinism: c_int,
    math_type: c_int,
    reserved: [3]c_int,
};

const Api = struct {
    cudnnCreate: *const fn (handle: *Handle) callconv(.c) Status,
    cudnnDestroy: *const fn (handle: Handle) callconv(.c) Status,
    cudnnSetStream: *const fn (handle: Handle, stream: Handle) callconv(.c) Status,
    cudnnGetVersion: *const fn () callconv(.c) usize,
    cudnnGetErrorString: *const fn (status: Status) callconv(.c) [*:0]const u8,
    cudnnCreateTensorDescriptor: *const fn (desc: *Handle) callconv(.c) Status,
    cudnnSetTensor4dDescriptor: *const fn (desc: Handle, format: c_int, data: c_int, n: c_int, c: c_int, h: c_int, w: c_int) callconv(.c) Status,
    cudnnDestroyTensorDescriptor: *const fn (desc: Handle) callconv(.c) Status,
    cudnnCreateFilterDescriptor: *const fn (desc: *Handle) callconv(.c) Status,
    cudnnSetFilter4dDescriptor: *const fn (desc: Handle, data: c_int, format: c_int, k: c_int, c: c_int, h: c_int, w: c_int) callconv(.c) Status,
    cudnnDestroyFilterDescriptor: *const fn (desc: Handle) callconv(.c) Status,
    cudnnCreateConvolutionDescriptor: *const fn (desc: *Handle) callconv(.c) Status,
    cudnnSetConvolution2dDescriptor: *const fn (desc: Handle, pad_h: c_int, pad_w: c_int, stride_h: c_int, stride_w: c_int, dilation_h: c_int, dilation_w: c_int, mode: c_int, compute: c_int) callconv(.c) Status,
    cudnnSetConvolutionMathType: *const fn (desc: Handle, math: c_int) callconv(.c) Status,
    cudnnDestroyConvolutionDescriptor: *const fn (desc: Handle) callconv(.c) Status,
    cudnnFindConvolutionForwardAlgorithm: *const fn (handle: Handle, x: Handle, w: Handle, conv: Handle, y: Handle, requested: c_int, returned: *c_int, results: [*]Perf) callconv(.c) Status,
    cudnnFindConvolutionBackwardDataAlgorithm: *const fn (handle: Handle, w: Handle, dy: Handle, conv: Handle, dx: Handle, requested: c_int, returned: *c_int, results: [*]Perf) callconv(.c) Status,
    cudnnFindConvolutionBackwardFilterAlgorithm: *const fn (handle: Handle, x: Handle, dy: Handle, conv: Handle, dw: Handle, requested: c_int, returned: *c_int, results: [*]Perf) callconv(.c) Status,
    cudnnConvolutionForward: *const fn (handle: Handle, alpha: *const f32, x_desc: Handle, x: ?*const anyopaque, w_desc: Handle, w: ?*const anyopaque, conv: Handle, algo: c_int, workspace: ?*anyopaque, workspace_bytes: usize, beta: *const f32, y_desc: Handle, y: ?*anyopaque) callconv(.c) Status,
    cudnnConvolutionBackwardData: *const fn (handle: Handle, alpha: *const f32, w_desc: Handle, w: ?*const anyopaque, dy_desc: Handle, dy: ?*const anyopaque, conv: Handle, algo: c_int, workspace: ?*anyopaque, workspace_bytes: usize, beta: *const f32, dx_desc: Handle, dx: ?*anyopaque) callconv(.c) Status,
    cudnnConvolutionBackwardFilter: *const fn (handle: Handle, alpha: *const f32, x_desc: Handle, x: ?*const anyopaque, dy_desc: Handle, dy: ?*const anyopaque, conv: Handle, algo: c_int, workspace: ?*anyopaque, workspace_bytes: usize, beta: *const f32, dw_desc: Handle, dw: ?*anyopaque) callconv(.c) Status,
};

pub const Geometry = struct {
    n: usize,
    h: usize,
    w: usize,
    c: usize,
    k: usize,
    r: usize,
    pad: usize,
    stride: usize,
};

const Choice = struct { algo: c_int, workspace: usize };

pub const Plan = struct {
    x: Handle,
    y: Handle,
    filter: Handle,
    conv: Handle,
    forward: Choice,
    backward_data: Choice,
    backward_filter: Choice,

    pub fn workspaceBytes(self: Plan) usize {
        return @max(self.forward.workspace, @max(self.backward_data.workspace, self.backward_filter.workspace));
    }
};

pub const Cudnn = struct {
    lib: dynlib.Library,
    api: Api,
    handle: Handle,

    pub fn open(dir: [:0]const u8) Error!Cudnn {
        const lib = dynlib.Library.open(.{ .windows = "cudnn64_9.dll", .linux = &.{ "libcudnn.so.9", "libcudnn.so" } }, dir) catch return error.CudnnNotFound;
        errdefer lib.close();
        const api = lib.bind(Api) catch return error.MissingSymbol;
        var self: Cudnn = .{ .lib = lib, .api = api, .handle = null };
        try self.check(api.cudnnCreate(&self.handle));
        return self;
    }

    pub fn close(self: *Cudnn) void {
        _ = self.api.cudnnDestroy(self.handle);
        self.lib.close();
    }

    /// Makes the calls of this handle go to stream (null: the default stream), which a graph recording needs.
    pub fn setStream(self: *const Cudnn, stream: Handle) Error!void {
        try self.check(self.api.cudnnSetStream(self.handle, stream));
    }

    pub fn version(self: *const Cudnn) usize {
        return self.api.cudnnGetVersion();
    }

    fn check(self: *const Cudnn, status: Status) Error!void {
        if (status == 0) return;
        std.debug.print("cuDNN call failed: {d} ({s})\n", .{ status, self.api.cudnnGetErrorString(status) });
        return error.CudnnFailed;
    }

    fn fastest(results: []const Perf, count: usize, workspace_limit: usize, deterministic: bool) Error!Choice {
        const deterministic_algorithm = 1;
        for (results[0..count]) |r| {
            if (r.status == 0 and r.memory <= workspace_limit and (!deterministic or r.determinism == deterministic_algorithm)) return .{ .algo = r.algo, .workspace = r.memory };
        }
        return error.NoAlgorithm;
    }

    pub fn plan(self: *Cudnn, g: Geometry, data: DataType, workspace_limit: usize, deterministic: bool) Error!Plan {
        const api = self.api;
        const out_h: c_int = @intCast((g.h + 2 * g.pad - g.r) / g.stride + 1);
        const out_w: c_int = @intCast((g.w + 2 * g.pad - g.r) / g.stride + 1);
        var p: Plan = undefined;
        const nhwc = 1;
        try self.check(api.cudnnCreateTensorDescriptor(&p.x));
        try self.check(api.cudnnSetTensor4dDescriptor(p.x, nhwc, @backingInt(data), @intCast(g.n), @intCast(g.c), @intCast(g.h), @intCast(g.w)));
        try self.check(api.cudnnCreateTensorDescriptor(&p.y));
        try self.check(api.cudnnSetTensor4dDescriptor(p.y, nhwc, @backingInt(data), @intCast(g.n), @intCast(g.k), out_h, out_w));
        try self.check(api.cudnnCreateFilterDescriptor(&p.filter));
        try self.check(api.cudnnSetFilter4dDescriptor(p.filter, @backingInt(data), nhwc, @intCast(g.k), @intCast(g.c), @intCast(g.r), @intCast(g.r)));
        try self.check(api.cudnnCreateConvolutionDescriptor(&p.conv));
        const cross_correlation = 1;
        try self.check(api.cudnnSetConvolution2dDescriptor(p.conv, @intCast(g.pad), @intCast(g.pad), @intCast(g.stride), @intCast(g.stride), 1, 1, cross_correlation, 0));
        const tensor_op_allow_conversion = 2;
        const tensor_op = 1;
        try self.check(api.cudnnSetConvolutionMathType(p.conv, if (data == .float) tensor_op_allow_conversion else tensor_op));

        var results: [16]Perf = undefined;
        var count: c_int = 0;
        try self.check(api.cudnnFindConvolutionForwardAlgorithm(self.handle, p.x, p.filter, p.conv, p.y, results.len, &count, &results));
        p.forward = try fastest(&results, @intCast(count), workspace_limit, deterministic);
        try self.check(api.cudnnFindConvolutionBackwardDataAlgorithm(self.handle, p.filter, p.y, p.conv, p.x, results.len, &count, &results));
        p.backward_data = try fastest(&results, @intCast(count), workspace_limit, deterministic);
        try self.check(api.cudnnFindConvolutionBackwardFilterAlgorithm(self.handle, p.x, p.y, p.conv, p.filter, results.len, &count, &results));
        p.backward_filter = try fastest(&results, @intCast(count), workspace_limit, deterministic);
        return p;
    }

    pub fn destroyPlan(self: *Cudnn, p: Plan) void {
        _ = self.api.cudnnDestroyTensorDescriptor(p.x);
        _ = self.api.cudnnDestroyTensorDescriptor(p.y);
        _ = self.api.cudnnDestroyFilterDescriptor(p.filter);
        _ = self.api.cudnnDestroyConvolutionDescriptor(p.conv);
    }

    pub fn forward(self: *Cudnn, p: Plan, x: u64, w: u64, y: u64, workspace: u64, accumulate: bool) Error!void {
        const alpha: f32 = 1;
        const beta: f32 = if (accumulate) 1 else 0;
        try self.check(self.api.cudnnConvolutionForward(self.handle, &alpha, p.x, @ptrFromInt(x), p.filter, @ptrFromInt(w), p.conv, p.forward.algo, @ptrFromInt(workspace), p.forward.workspace, &beta, p.y, @ptrFromInt(y)));
    }

    pub fn backwardData(self: *Cudnn, p: Plan, w: u64, dy: u64, dx: u64, workspace: u64, accumulate: bool) Error!void {
        const alpha: f32 = 1;
        const beta: f32 = if (accumulate) 1 else 0;
        try self.check(self.api.cudnnConvolutionBackwardData(self.handle, &alpha, p.filter, @ptrFromInt(w), p.y, @ptrFromInt(dy), p.conv, p.backward_data.algo, @ptrFromInt(workspace), p.backward_data.workspace, &beta, p.x, @ptrFromInt(dx)));
    }

    pub fn backwardFilter(self: *Cudnn, p: Plan, x: u64, dy: u64, dw: u64, workspace: u64, accumulate: bool) Error!void {
        const alpha: f32 = 1;
        const beta: f32 = if (accumulate) 1 else 0;
        try self.check(self.api.cudnnConvolutionBackwardFilter(self.handle, &alpha, p.x, @ptrFromInt(x), p.y, @ptrFromInt(dy), p.conv, p.backward_filter.algo, @ptrFromInt(workspace), p.backward_filter.workspace, &beta, p.filter, @ptrFromInt(dw)));
    }
};
