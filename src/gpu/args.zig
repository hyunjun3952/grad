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
    first_block: [max_tensors]u32,
    w: [max_tensors]u64,
    g: [max_tensors]u64,
    m: [max_tensors]u64,
    v: [max_tensors]u64,
};

pub const AdamArgs = extern struct { beta1: f32, beta2: f32, step: f32, inv_sqrt_c2: f32, eps: f32, decay: f32 };

pub const max_rank = 6;

pub const ColumnArgs = extern struct { a: [*]const f32, b: [*]const f32, mean: [*]const f32, inv_std: [*]const f32, mask: [*]const f32, out: [*]f32, partial: [*]f32, ordered: u32, rows: u32, channels: u32, row_stride: u32, kind: u32, scale: f32 };

pub const ElementwiseArgs = extern struct { len: u32, multiply: u32, dims: [max_rank]u32, a_strides: [max_rank]u32, b_strides: [max_rank]u32 };

pub const SumArgs = extern struct { out_len: u32, reduce_count: u32, mask: u32, accumulate: u32, dims: [max_rank]u32, strides: [max_rank]u32 };

pub const GatherArgs = extern struct { start: u32, rows: u32, pixels: u32, max_shift: u32, max_angle: f32, max_scale: f32, channels: u32, flip: u32, mirror: u32, cutout: f32, x0: u32, y0: u32, x1: u32, y1: u32, seed: u64 };

pub const PackArgs = extern struct { rows: u32, cols: u32, rows_padded: u32, cols_padded: u32, src_stride: u32, transpose: u32 };

pub const TensorShape = extern struct { m: u32, n: u32, k_padded: u32, n_padded: u32, accumulate: u32, chunk: u32, transpose_out: u32, output_half: u32, split_stride: u32 };

pub const ConvGeometry = extern struct { n: u32, h: u32, w: u32, c: u32, k: u32, stride: u32, pad: u32, out_h: u32, out_w: u32, positions: u32, c_out: u32 };

pub const RotateArgs = extern struct { k: u32, c_in: u32, c_out: u32, rows_padded: u32, cols_padded: u32 };

pub const LayoutArgs = extern struct { r: u32, c: u32, k: u32 };
