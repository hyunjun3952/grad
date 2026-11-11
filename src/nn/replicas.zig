const std = @import("std");
const Allocator = std.mem.Allocator;
const kernels = @import("../cpu/kernels.zig");

// ***** data parallel training *****

/// Synchronisation between model replicas that train on slices of one batch, one replica per thread.
/// The gradients are averaged through host memory: every replica copies its gradients out, the sum is formed in
/// parallel, and every replica copies the mean back, so all replicas take the same optimizer step.
pub const Replicas = struct {
    gpa: Allocator,
    count: usize,
    host: [][]f32,
    arrived: std.atomic.Value(u32) = .init(0),
    generation: std.atomic.Value(u32) = .init(0),
    stopped: std.atomic.Value(bool) = .init(false),

    /// Allocates one host buffer of len floats per replica; len is the total number of gradient elements.
    pub fn init(gpa: Allocator, count: usize, len: usize) Allocator.Error!Replicas {
        const host = try gpa.alloc([]f32, count);
        var made: usize = 0;
        errdefer {
            for (host[0..made]) |h| gpa.free(h);
            gpa.free(host);
        }
        while (made < count) : (made += 1) host[made] = try gpa.alloc(f32, len);
        return .{ .gpa = gpa, .count = count, .host = host };
    }

    pub fn deinit(self: *Replicas) void {
        for (self.host) |h| self.gpa.free(h);
        self.gpa.free(self.host);
    }

    /// Releases every replica that waits in a barrier with error.Stopped. A replica that fails calls this so that the others do not hang.
    pub fn stop(self: *Replicas) void {
        self.stopped.store(true, .release);
    }

    /// Blocks until all replicas have called it. Returns error.Stopped if another replica failed.
    pub fn wait(self: *Replicas) error{Stopped}!void {
        if (self.stopped.load(.acquire)) return error.Stopped;
        const generation = self.generation.load(.acquire);
        if (self.arrived.fetchAdd(1, .acq_rel) + 1 == self.count) {
            self.arrived.store(0, .monotonic);
            _ = self.generation.fetchAdd(1, .release);
            return;
        }
        while (self.generation.load(.acquire) == generation) {
            if (self.stopped.load(.acquire)) return error.Stopped;
            std.Thread.yield() catch {};
        }
    }

    /// Replaces the gradient of every parameter, in every replica, by the mean over the replicas. Every replica calls it with its own parameters.
    /// NOTE: the sum runs in replica order, so all replicas end with identical bits and stay identical after the optimizer step.
    pub fn averageGrads(self: *Replicas, index: usize, par: kernels.Parallel, params: anytype) !void {
        if (self.count == 1) return;
        var at: usize = 0;
        inline for (params) |p| {
            const grad = p.grad.data;
            const mine = self.host[index][at..][0..grad.len];
            try par.download(mine, grad);
            at += grad.len;
        }
        std.debug.assert(at == self.host[0].len);
        try self.wait();

        const len = self.host[0].len;
        const scale = 1 / @as(f32, @floatFromInt(self.count));
        for (len * index / self.count..len * (index + 1) / self.count) |i| {
            var sum: f32 = 0;
            for (self.host) |h| sum += h[i];
            self.host[0][i] = sum * scale;
        }
        try self.wait();

        at = 0;
        inline for (params) |p| {
            const grad = p.grad.data;
            const mean = self.host[0][at..][0..grad.len];
            try par.upload(grad, mean);
            at += grad.len;
        }
        // NOTE: the next call overwrites host[0], so no replica may still be reading it.
        try self.wait();
    }
};

/// Returns the total number of gradient elements of a parameter tuple, which is the len that Replicas.init needs.
pub fn gradLen(params: anytype) usize {
    var total: usize = 0;
    inline for (params) |p| total += p.grad.data.len;
    return total;
}
