const drop = @import("dropout.zig");

pub const Box = struct {
    x0: u32 = 0,
    y0: u32 = 0,
    x1: u32 = 0,
    y1: u32 = 0,

    pub inline fn contains(self: Box, index: usize, width: usize, channels: usize) bool {
        const x = (index / channels) % width;
        const y = (index / channels) / width;
        return x >= self.x0 and x < self.x1 and y >= self.y0 and y < self.y1;
    }
};

pub const Config = struct {
    max_shift: u32 = 0,
    max_angle: f32 = 0,
    max_scale: f32 = 0,
    flip: bool = false,
    mirror: bool = false,
    cutout: f32 = 0,
    box: Box = .{},
    channels: u32 = 1,
    seed: u64 = 0,

    pub inline fn active(self: Config) bool {
        return self.max_shift != 0 or self.max_angle != 0 or self.max_scale != 0 or self.flip or self.mirror or self.cutout != 0;
    }
};

pub const Transform = struct { a: f32, b: f32, c: f32, d: f32, dx: f32, dy: f32, cut: f32 = 0, cut_x: f32 = 0, cut_y: f32 = 0 };

pub const identity: Transform = .{ .a = 1, .b = 0, .c = 0, .d = 1, .dx = 0, .dy = 0 };

inline fn unit(bits: u64) f32 {
    return @as(f32, @floatFromInt(bits >> 40)) * (2.0 / 16777216.0) - 1;
}

inline fn sine(x: f32) f32 {
    const x2 = x * x;
    return x * (1 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0)))));
}

inline fn cosine(x: f32) f32 {
    const x2 = x * x;
    return 1 + x2 * (-1.0 / 2.0 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0))));
}

pub inline fn transformOf(sample: usize, config: Config) Transform {
    const span: u64 = 2 * @as(u64, config.max_shift) + 1;
    const shift_bits = drop.hash(config.seed, 4 * sample);
    const dx = @as(i32, @intCast(shift_bits % span)) - @as(i32, @intCast(config.max_shift));
    const dy = @as(i32, @intCast((shift_bits >> 32) % span)) - @as(i32, @intCast(config.max_shift));
    // Degrees to radians; the polynomials above stay accurate up to about 45 degrees.
    const angle = unit(drop.hash(config.seed, 4 * sample + 1)) * config.max_angle * (3.14159265 / 180.0);
    const zoom = 1 + unit(drop.hash(config.seed, 4 * sample + 2)) * config.max_scale;
    const mirror: f32 = if (config.mirror or (config.flip and drop.hash(config.seed, 4 * sample + 3) & 1 == 1)) -1 else 1;
    const cut_bits = drop.hash(config.seed, 4 * sample + 3) >> 8;
    return .{
        .a = mirror * cosine(angle) / zoom,
        .b = sine(angle) / zoom,
        .c = -mirror * sine(angle) / zoom,
        .d = cosine(angle) / zoom,
        .dx = @floatFromInt(dx),
        .dy = @floatFromInt(dy),
        .cut = config.cutout,
        .cut_x = @as(f32, @floatFromInt(cut_bits & 0xFFFFFF)) / 16777216.0,
        .cut_y = @as(f32, @floatFromInt((cut_bits >> 24) & 0xFFFFFF)) / 16777216.0,
    };
}

inline fn at(image: [*]const f32, x: i32, y: i32, width: i32, channels: usize, channel: usize) f32 {
    return if (x < 0 or y < 0 or x >= width or y >= width) 0 else image[(@as(usize, @intCast(y)) * @as(usize, @intCast(width)) + @as(usize, @intCast(x))) * channels + channel];
}

pub inline fn pixel(image: [*]const f32, index: usize, width: usize, channels: usize, tr: Transform) f32 {
    const p = index / channels;
    const channel = index % channels;
    const w: i32 = @intCast(width);
    const wf: f32 = @floatFromInt(width);
    if (tr.cut > 0 and @abs(@as(f32, @floatFromInt(p % width)) + 0.5 - tr.cut_x * wf) < tr.cut / 2 and @abs(@as(f32, @floatFromInt(p / width)) + 0.5 - tr.cut_y * wf) < tr.cut / 2) return 0;
    const middle = (@as(f32, @floatFromInt(width)) - 1) / 2;
    const ox = @as(f32, @floatFromInt(p % width)) - middle;
    const oy = @as(f32, @floatFromInt(p / width)) - middle;
    const sx = tr.a * ox + tr.b * oy + middle - tr.dx;
    const sy = tr.c * ox + tr.d * oy + middle - tr.dy;
    const x0: i32 = @intFromFloat(@floor(sx));
    const y0: i32 = @intFromFloat(@floor(sy));
    const fx = sx - @as(f32, @floatFromInt(x0));
    const fy = sy - @as(f32, @floatFromInt(y0));
    const top = at(image, x0, y0, w, channels, channel) * (1 - fx) + at(image, x0 + 1, y0, w, channels, channel) * fx;
    const bottom = at(image, x0, y0 + 1, w, channels, channel) * (1 - fx) + at(image, x0 + 1, y0 + 1, w, channels, channel) * fx;
    return top * (1 - fy) + bottom * fy;
}

pub inline fn side(floats: usize, channels: usize) usize {
    return @intFromFloat(@sqrt(@as(f32, @floatFromInt(floats / channels))));
}
