pub inline fn encode(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    // Round to nearest even: add half of the dropped 16 bits, plus one when the kept lowest bit is set.
    return @truncate((bits +% 0x7FFF +% ((bits >> 16) & 1)) >> 16);
}

pub inline fn load(comptime E: type, p: [*]const E, i: usize) f32 {
    return if (E == u16) decode(p[i]) else p[i];
}

pub inline fn store(comptime E: type, p: [*]E, i: usize, v: f32) void {
    p[i] = if (E == u16) encode(v) else v;
}

pub const Quad = @Vector(4, f32);

pub inline fn loadQuad(comptime E: type, p: [*]const E, i: usize) Quad {
    if (E == u16) {
        const raw: *align(8) const [4]u16 = @ptrCast(@alignCast(p + i));
        const bits: @Vector(4, u32) = @as(@Vector(4, u16), raw.*);
        return @bitCast(bits << @as(@Vector(4, u5), @splat(16)));
    }
    const raw: *align(16) const [4]f32 = @ptrCast(@alignCast(p + i));
    return raw.*;
}

pub inline fn storeQuad(comptime E: type, p: [*]E, i: usize, v: Quad) void {
    if (E == u16) {
        const bits: @Vector(4, u32) = @bitCast(v);
        const rounded = (bits +% @as(@Vector(4, u32), @splat(0x7FFF)) +% ((bits >> @as(@Vector(4, u5), @splat(16))) & @as(@Vector(4, u32), @splat(1)))) >> @as(@Vector(4, u5), @splat(16));
        const raw: *align(8) [4]u16 = @ptrCast(@alignCast(p + i));
        raw.* = @as(@Vector(4, u16), @truncate(rounded));
        return;
    }
    const raw: *align(16) [4]f32 = @ptrCast(@alignCast(p + i));
    raw.* = v;
}

pub inline fn decode(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}
