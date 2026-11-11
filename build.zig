const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.option(std.lang.Optimize, "optimize", "Optimization mode of the program (default: fast)") orelse .fast;

    const mod = b.addModule("grad", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const gpu_kernels = b.addObject(.{
        .name = "gpu_kernels",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/device.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_86 },
            }),
            .optimize = .fast,
        }),
    });
    mod.addAnonymousImport("kernels_ptx", .{ .root_source_file = gpu_kernels.getEmittedAsm() });

    const config = b.addOptions();
    config.addOption(usize, "hidden", b.option(usize, "hidden", "mnist example: hidden units of the MLP (default: 128)") orelse 128);
    config.addOption(usize, "batch", b.option(usize, "batch", "mnist example: training batch size (default: 64)") orelse 64);
    config.addOption(bool, "cnn", b.option(bool, "cnn", "mnist example: train the convolutional network instead of the MLP") orelse false);
    config.addOption(bool, "wide", b.option(bool, "wide", "cifar example: widths 64-192-384 instead of 64-128-256 (slower, more accurate)") orelse false);

    config.addOption(bool, "cifar100", b.option(bool, "cifar100", "cifar example: train on CIFAR-100 (data/cifar100) instead of CIFAR-10") orelse false);

    inline for (.{ "mnist", "cifar", "regression", "charlm", "onnx" }) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path("examples/" ++ name ++ "/main.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "grad", .module = mod },
                },
            }),
        });
        exe.root_module.addOptions("config", config);
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        run_cmd.addPassthruArgs();
        b.step(name, "Train the " ++ name ++ " example (zig build " ++ name ++ " -- [epochs] [--gpu] ...)").dependOn(&run_cmd.step);
    }
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .fast,
            .imports = &.{
                .{ .name = "grad", .module = mod },
            },
        }),
    });
    const bench_run = b.addRunArtifact(bench_exe);
    bench_run.addPassthruArgs();
    const bench_step = b.step("bench", "Run the matmul benchmark (ReleaseFast)");
    bench_step.dependOn(&bench_run.step);
}
