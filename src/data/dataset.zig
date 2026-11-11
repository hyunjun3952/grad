const std = @import("std");
const Allocator = std.mem.Allocator;
const Gpu = @import("../gpu/gpu.zig").Gpu;

pub const Dataset = struct {
    pixels: usize,
    images: []f32,
    labels: []u8,

    pub fn count(self: Dataset) usize {
        return self.labels.len;
    }

    pub fn toGpu(self: Dataset, gpu: *Gpu) !Dataset {
        const images = try gpu.alloc(f32, self.images.len);
        const labels = try gpu.alloc(u8, self.labels.len);
        try gpu.upload(images, self.images);
        try gpu.upload(labels, self.labels);
        return .{ .pixels = self.pixels, .images = images, .labels = labels };
    }

    pub fn deinit(self: Dataset, gpa: Allocator) void {
        gpa.free(self.images);
        gpa.free(self.labels);
    }
};
