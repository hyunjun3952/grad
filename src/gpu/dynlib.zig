const std = @import("std");
const builtin = @import("builtin");

const windows = builtin.os.tag == .windows;

const win = struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn SetDllDirectoryA(path: [*:0]const u8) callconv(.winapi) c_int;
    extern "kernel32" fn GetProcAddress(module: ?*anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(module: ?*anyopaque) callconv(.winapi) c_int;
};

const posix = struct {
    const now_and_global = 0x102;
    extern "c" fn dlopen(path: [*:0]const u8, flags: c_int) ?*anyopaque;
    extern "c" fn dlsym(handle: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
    extern "c" fn dlclose(handle: ?*anyopaque) c_int;
};

pub const Library = struct {
    handle: ?*anyopaque,

    pub const Names = struct {
        windows: [:0]const u8,
        linux: []const [:0]const u8,
    };

    pub fn open(names: Names, dir: ?[:0]const u8) error{LibraryNotFound}!Library {
        if (windows) {
            if (dir) |d| _ = win.SetDllDirectoryA(d.ptr);
            return .{ .handle = win.LoadLibraryA(names.windows.ptr) orelse return error.LibraryNotFound };
        }
        var path: [512]u8 = undefined;
        for (names.linux) |name| {
            if (dir) |d| {
                const text = std.fmt.bufPrint(&path, "{s}/{s}\x00", .{ d, name }) catch continue;
                if (posix.dlopen(@ptrCast(text.ptr), posix.now_and_global)) |handle| return .{ .handle = handle };
            }
            if (posix.dlopen(name.ptr, posix.now_and_global)) |handle| return .{ .handle = handle };
        }
        return error.LibraryNotFound;
    }

    pub fn symbol(self: Library, name: [:0]const u8) ?*anyopaque {
        return if (windows) win.GetProcAddress(self.handle, name.ptr) else posix.dlsym(self.handle, name.ptr);
    }

    pub fn close(self: Library) void {
        if (windows) _ = win.FreeLibrary(self.handle) else _ = posix.dlclose(self.handle);
    }

    pub fn bind(self: Library, comptime Api: type) error{MissingSymbol}!Api {
        var api: Api = undefined;
        inline for (@typeInfo(Api).@"struct".field_names) |name| {
            @field(api, name) = @ptrCast(self.symbol(name) orelse return error.MissingSymbol);
        }
        return api;
    }
};
