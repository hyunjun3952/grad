pub const GemmShape = extern struct {
    m: u32,
    n: u32,
    k: u32,
    accumulate: u32,
    chunk: u32,
};

pub const max_tensors = 16;

pub const TensorTable = extern struct {
    count: u32,
    len: [max_tensors]u32,
    w: [max_tensors]u64,
    g: [max_tensors]u64,
    m: [max_tensors]u64,
    v: [max_tensors]u64,
};

pub const AdamArgs = extern struct { beta1: f32, beta2: f32, step: f32, inv_sqrt_c2: f32, eps: f32, decay: f32 };
