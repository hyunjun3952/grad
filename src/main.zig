const std = @import("std");
const Io = std.Io;

const grad = @import("grad");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    const a = try grad.Tensor(f32, &.{ 2, 3 }).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit(gpa);
    const b = try grad.Tensor(f32, &.{ 3, 2 }).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit(gpa);
    const bias = try grad.Tensor(f32, &.{2}).fromSlice(gpa, &.{ 0.5, -0.5 });
    defer bias.deinit(gpa);

    const c = try grad.matmul(gpa, .{}, a, b, .{});
    defer c.deinit(gpa);
    const d = try grad.ops.add(gpa, c, bias);
    defer d.deinit(gpa);
    const p = try grad.ops.softmax(gpa, d);
    defer p.deinit(gpa);

    try out.print("a * b          = {any}\n", .{c.data});
    try out.print("a * b + bias   = {any}\n", .{d.data});
    try out.print("softmax(rows)  = {any}\n", .{p.data});
    try out.flush();
}
