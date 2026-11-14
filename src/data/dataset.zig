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

    /// Fails with LabelOutOfRange when a label is not below classes; run it on host data before training, because the loss indexes by label.
    pub fn checkLabels(self: Dataset, classes: usize) error{LabelOutOfRange}!void {
        for (self.labels) |label| {
            if (label >= classes) return error.LabelOutOfRange;
        }
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
