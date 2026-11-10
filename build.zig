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
            .root_source_file = b.path("src/gpu/kernels.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_86 },
            }),
            .optimize = .fast,
        }),
    });
    mod.addAnonymousImport("kernels_ptx", .{ .root_source_file = gpu_kernels.getEmittedAsm() });

    const exe = b.addExecutable(.{
        .name = "grad",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "grad", .module = mod },
            },
        }),
    });

    const config = b.addOptions();
    config.addOption(usize, "hidden", b.option(usize, "hidden", "Hidden units of the trained MLP (default: 128)") orelse 128);
    config.addOption(usize, "batch", b.option(usize, "batch", "Training batch size (default: 64)") orelse 64);
    exe.root_module.addOptions("config", config);

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    run_cmd.addPassthruArgs();

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
