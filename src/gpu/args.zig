pub const GemmShape = extern struct {
    m: u32,
    n: u32,
    k: u32,
    accumulate: u32,
    chunk: u32,
};

pub const max_tensors = 64;

pub const TensorTable = extern struct {
    count: u32,
    len: [max_tensors]u32,
    w: [max_tensors]u64,
    g: [max_tensors]u64,
    m: [max_tensors]u64,
    v: [max_tensors]u64,
};

pub const AdamArgs = extern struct { beta1: f32, beta2: f32, step: f32, inv_sqrt_c2: f32, eps: f32, decay: f32 };

pub const ElementwiseArgs = extern struct { len: u32, multiply: u32, dims: [4]u32, a_strides: [4]u32, b_strides: [4]u32 };

pub const SumArgs = extern struct { out_len: u32, reduce_count: u32, mask: u32, accumulate: u32, dims: [4]u32, strides: [4]u32 };

pub const GatherArgs = extern struct { start: u32, rows: u32, pixels: u32, max_shift: u32, max_angle: f32, max_scale: f32, channels: u32, flip: u32, seed: u64 };

pub const PackArgs = extern struct { rows: u32, cols: u32, rows_padded: u32, cols_padded: u32, src_stride: u32, transpose: u32 };

pub const TensorShape = extern struct { m: u32, n: u32, k_padded: u32, n_padded: u32, accumulate: u32, chunk: u32, transpose_out: u32 };
