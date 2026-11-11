const std = @import("std");
const Allocator = std.mem.Allocator;
const dynlib = @import("dynlib.zig");

pub const DevicePtr = u64;
pub const Handle = ?*anyopaque;
const Status = c_int;

const Api = struct {
    cuInit: *const fn (flags: c_uint) callconv(.c) Status,
    cuDeviceGet: *const fn (device: *c_int, ordinal: c_int) callconv(.c) Status,
    cuDevicePrimaryCtxRetain: *const fn (ctx: *Handle, device: c_int) callconv(.c) Status,
    cuCtxSetCurrent: *const fn (ctx: Handle) callconv(.c) Status,
    cuCtxSynchronize: *const fn () callconv(.c) Status,
    cuMemAlloc_v2: *const fn (ptr: *DevicePtr, bytes: usize) callconv(.c) Status,
    cuMemFree_v2: *const fn (ptr: DevicePtr) callconv(.c) Status,
    cuMemsetD32_v2: *const fn (dst: DevicePtr, value: c_uint, count: usize) callconv(.c) Status,
    cuMemcpyHtoD_v2: *const fn (dst: DevicePtr, src: *const anyopaque, bytes: usize) callconv(.c) Status,
    cuMemcpyDtoH_v2: *const fn (dst: *anyopaque, src: DevicePtr, bytes: usize) callconv(.c) Status,
    cuModuleLoadData: *const fn (module: *Handle, image: *const anyopaque) callconv(.c) Status,
    cuModuleGetFunction: *const fn (func: *Handle, module: Handle, name: [*:0]const u8) callconv(.c) Status,
    cuLaunchKernel: *const fn (
        f: Handle,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        shared_bytes: c_uint,
        stream: Handle,
        params: [*]?*anyopaque,
        extra: ?[*]?*anyopaque,
    ) callconv(.c) Status,
    cuGetErrorName: *const fn (code: Status, name: *[*:0]const u8) callconv(.c) Status,
};

pub const Cuda = struct {
    lib: dynlib.Library,
    api: Api,
    device: c_int,

    pub fn open() !Cuda {
        const lib = dynlib.Library.open(.{ .windows = "nvcuda.dll", .linux = &.{ "libcuda.so.1", "libcuda.so" } }, null) catch return error.DriverNotFound;
        errdefer lib.close();
        const api = try lib.bind(Api);

        var self: Cuda = .{ .lib = lib, .api = api, .device = 0 };
        try self.check(api.cuInit(0));
        try self.check(api.cuDeviceGet(&self.device, 0));
        var ctx: Handle = null;
        try self.check(api.cuDevicePrimaryCtxRetain(&ctx, self.device));
        try self.check(api.cuCtxSetCurrent(ctx));
        return self;
    }

    pub fn close(self: *Cuda) void {
        self.lib.close();
    }

    fn check(self: *const Cuda, status: Status) !void {
        if (status == 0) return;
        var name: [*:0]const u8 = "unknown";
        _ = self.api.cuGetErrorName(status, &name);
        std.debug.print("CUDA driver call failed: {d} ({s})\n", .{ status, name });
        return error.CudaFailed;
    }

    pub fn alloc(self: *const Cuda, bytes: usize) !DevicePtr {
        var ptr: DevicePtr = 0;
        try self.check(self.api.cuMemAlloc_v2(&ptr, bytes));
        return ptr;
    }

    pub fn free(self: *const Cuda, ptr: DevicePtr) void {
        _ = self.api.cuMemFree_v2(ptr);
    }

    pub fn memsetD32(self: *const Cuda, dst: DevicePtr, value: u32, count: usize) !void {
        try self.check(self.api.cuMemsetD32_v2(dst, value, count));
    }

    pub fn upload(self: *const Cuda, dst: DevicePtr, src: anytype) !void {
        const bytes = std.mem.sliceAsBytes(src);
        try self.check(self.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len));
    }

    pub fn download(self: *const Cuda, dst: anytype, src: DevicePtr) !void {
        const bytes = std.mem.sliceAsBytes(dst);
        try self.check(self.api.cuMemcpyDtoH_v2(bytes.ptr, src, bytes.len));
    }

    pub fn loadModule(self: *const Cuda, gpa: Allocator, ptx: []const u8) !Handle {
        const text = try gpa.alloc(u8, ptx.len + 1);
        defer gpa.free(text);
        @memcpy(text[0..ptx.len], ptx);
        text[ptx.len] = 0;
        var module: Handle = null;
        try self.check(self.api.cuModuleLoadData(&module, text.ptr));
        return module;
    }

    pub fn getFunction(self: *const Cuda, module: Handle, name: [*:0]const u8) !Handle {
        var f: Handle = null;
        try self.check(self.api.cuModuleGetFunction(&f, module, name));
        return f;
    }

    pub fn launch(self: *const Cuda, f: Handle, grid: [3]u32, block: [3]u32, args: []?*anyopaque) !void {
        try self.check(self.api.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, null, args.ptr, null));
    }

    pub fn sync(self: *const Cuda) !void {
        try self.check(self.api.cuCtxSynchronize());
    }
};
