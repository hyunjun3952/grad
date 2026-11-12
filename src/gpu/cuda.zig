const std = @import("std");
const Allocator = std.mem.Allocator;
const dynlib = @import("dynlib.zig");

pub const DevicePtr = u64;
pub const Handle = ?*anyopaque;
const Status = c_int;

const Api = struct {
    cuInit: *const fn (flags: c_uint) callconv(.c) Status,
    cuDeviceGet: *const fn (device: *c_int, ordinal: c_int) callconv(.c) Status,
    cuDeviceGetCount: *const fn (count: *c_int) callconv(.c) Status,
    cuDeviceGetAttribute: *const fn (value: *c_int, attribute: c_int, device: c_int) callconv(.c) Status,
    cuDevicePrimaryCtxRetain: *const fn (ctx: *Handle, device: c_int) callconv(.c) Status,
    cuCtxSetCurrent: *const fn (ctx: Handle) callconv(.c) Status,
    cuCtxSynchronize: *const fn () callconv(.c) Status,
    cuMemAlloc_v2: *const fn (ptr: *DevicePtr, bytes: usize) callconv(.c) Status,
    cuMemFree_v2: *const fn (ptr: DevicePtr) callconv(.c) Status,
    cuMemsetD32_v2: *const fn (dst: DevicePtr, value: c_uint, count: usize) callconv(.c) Status,
    cuMemcpyHtoD_v2: *const fn (dst: DevicePtr, src: *const anyopaque, bytes: usize) callconv(.c) Status,
    cuMemcpyDtoH_v2: *const fn (dst: *anyopaque, src: DevicePtr, bytes: usize) callconv(.c) Status,
    cuModuleLoadData: *const fn (module: *Handle, image: *const anyopaque) callconv(.c) Status,
    cuModuleUnload: *const fn (module: Handle) callconv(.c) Status,
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

// Stream capture and graphs are bound separately, so a driver without them still runs everything else.
const GraphApi = struct {
    cuStreamCreate: *const fn (stream: *Handle, flags: c_uint) callconv(.c) Status,
    cuStreamDestroy_v2: *const fn (stream: Handle) callconv(.c) Status,
    cuStreamBeginCapture_v2: *const fn (stream: Handle, mode: c_int) callconv(.c) Status,
    cuStreamEndCapture: *const fn (stream: Handle, graph: *Handle) callconv(.c) Status,
    cuGraphInstantiateWithFlags: *const fn (exec: *Handle, graph: Handle, flags: c_ulonglong) callconv(.c) Status,
    cuGraphLaunch: *const fn (exec: Handle, stream: Handle) callconv(.c) Status,
    cuGraphDestroy: *const fn (graph: Handle) callconv(.c) Status,
    cuGraphExecDestroy: *const fn (exec: Handle) callconv(.c) Status,
    cuMemsetD32Async: *const fn (dst: DevicePtr, value: c_uint, count: usize, stream: Handle) callconv(.c) Status,
};

pub const Cuda = struct {
    lib: dynlib.Library,
    api: Api,
    graph_api: ?GraphApi,
    device: c_int,
    context: Handle,
    // The module that loadModule returned, unloaded by close.
    module: Handle = null,
    // The stream that launches go to: null (the default stream) normally, the capture stream while a graph is recorded.
    stream: Handle = null,
    capture_stream: Handle = null,

    /// Opens the driver and makes the primary context of the device with this ordinal current on the calling thread.
    /// NOTE: the current context is per thread, so a program that drives several devices must call makeCurrent from the thread that uses each one.
    pub fn open(ordinal: usize) !Cuda {
        const lib = dynlib.Library.open(.{ .windows = "nvcuda.dll", .linux = &.{ "libcuda.so.1", "libcuda.so" } }, null) catch return error.DriverNotFound;
        errdefer lib.close();
        const api = try lib.bind(Api);

        var self: Cuda = .{ .lib = lib, .api = api, .graph_api = lib.bind(GraphApi) catch null, .device = 0, .context = null };
        try self.check(api.cuInit(0));
        var count: c_int = 0;
        try self.check(api.cuDeviceGetCount(&count));
        if (ordinal >= count) return error.NoSuchDevice;
        try self.check(api.cuDeviceGet(&self.device, @intCast(ordinal)));
        try self.check(api.cuDevicePrimaryCtxRetain(&self.context, self.device));
        try self.check(api.cuCtxSetCurrent(self.context));
        return self;
    }

    /// Returns how many CUDA devices the driver reports, or zero when there is no driver.
    pub fn deviceCount() usize {
        const lib = dynlib.Library.open(.{ .windows = "nvcuda.dll", .linux = &.{ "libcuda.so.1", "libcuda.so" } }, null) catch return 0;
        defer lib.close();
        const api = lib.bind(Api) catch return 0;
        if (api.cuInit(0) != 0) return 0;
        var count: c_int = 0;
        if (api.cuDeviceGetCount(&count) != 0) return 0;
        return @intCast(count);
    }

    /// Returns the number of streaming multiprocessors of the device.
    pub fn multiprocessorCount(self: *const Cuda) !usize {
        const multiprocessor_count_attribute = 16;
        var count: c_int = 0;
        try self.check(self.api.cuDeviceGetAttribute(&count, multiprocessor_count_attribute, self.device));
        return @intCast(count);
    }

    /// Makes this device's context current on the calling thread.
    pub fn makeCurrent(self: *const Cuda) !void {
        try self.check(self.api.cuCtxSetCurrent(self.context));
    }

    /// Destroys the capture stream, unloads the module and closes the driver library. Device memory must be freed before.
    /// NOTE: the primary context stays retained on purpose. It is shared by every Gpu of the process, so keeping it alive spares
    /// the later ones from creating it again; the module is per Gpu and would otherwise pile up inside that context.
    pub fn close(self: *Cuda) void {
        if (self.capture_stream != null) _ = self.graph_api.?.cuStreamDestroy_v2(self.capture_stream);
        if (self.module != null) _ = self.api.cuModuleUnload(self.module);
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
        // NOTE: the synchronous memset is not allowed while a stream is being captured, so a recording uses the stream-ordered one.
        if (self.stream != null) return self.check(self.graph_api.?.cuMemsetD32Async(dst, value, count, self.stream));
        try self.check(self.api.cuMemsetD32_v2(dst, value, count));
    }

    /// Starts recording every launch and memset into a graph instead of running them. Fails when the driver has no graph support.
    pub fn beginCapture(self: *Cuda) !void {
        const g = self.graph_api orelse return error.NoGraphSupport;
        // NOTE: non-blocking (flag 1). Replicas on one device share the primary context and with it the legacy default stream; a
        // blocking stream is implicitly ordered against that stream, so work another thread puts on it would invalidate the recording.
        if (self.capture_stream == null) try self.check(g.cuStreamCreate(&self.capture_stream, 1));
        // Relaxed mode: other threads and the ordinary memory calls of the host are not policed.
        try self.check(g.cuStreamBeginCapture_v2(self.capture_stream, 2));
        self.stream = self.capture_stream;
    }

    /// Ends the recording and returns the instantiated graph, which replay can run.
    pub fn endCapture(self: *Cuda) !Handle {
        const g = self.graph_api.?;
        self.stream = null;
        var graph: Handle = null;
        try self.check(g.cuStreamEndCapture(self.capture_stream, &graph));
        defer _ = g.cuGraphDestroy(graph);
        var exec: Handle = null;
        try self.check(g.cuGraphInstantiateWithFlags(&exec, graph, 0));
        return exec;
    }

    pub fn replay(self: *const Cuda, exec: Handle) !void {
        try self.check(self.graph_api.?.cuGraphLaunch(exec, null));
    }

    pub fn destroyGraph(self: *const Cuda, exec: Handle) void {
        _ = self.graph_api.?.cuGraphExecDestroy(exec);
    }

    pub fn upload(self: *const Cuda, dst: DevicePtr, src: anytype) !void {
        const bytes = std.mem.sliceAsBytes(src);
        try self.check(self.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len));
    }

    pub fn download(self: *const Cuda, dst: anytype, src: DevicePtr) !void {
        const bytes = std.mem.sliceAsBytes(dst);
        try self.check(self.api.cuMemcpyDtoH_v2(bytes.ptr, src, bytes.len));
    }

    /// Loads the PTX image as the module of this Cuda, which close unloads. Call it once.
    pub fn loadModule(self: *Cuda, gpa: Allocator, ptx: []const u8) !Handle {
        std.debug.assert(self.module == null);
        const text = try gpa.alloc(u8, ptx.len + 1);
        defer gpa.free(text);
        @memcpy(text[0..ptx.len], ptx);
        text[ptx.len] = 0;
        var module: Handle = null;
        try self.check(self.api.cuModuleLoadData(&module, text.ptr));
        self.module = module;
        return module;
    }

    pub fn getFunction(self: *const Cuda, module: Handle, name: [*:0]const u8) !Handle {
        var f: Handle = null;
        try self.check(self.api.cuModuleGetFunction(&f, module, name));
        return f;
    }

    pub fn launch(self: *const Cuda, f: Handle, grid: [3]u32, block: [3]u32, args: []?*anyopaque) !void {
        try self.check(self.api.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, self.stream, args.ptr, null));
    }

    pub fn sync(self: *const Cuda) !void {
        try self.check(self.api.cuCtxSynchronize());
    }
};
