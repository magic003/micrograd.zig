const std = @import("std");
const Prng = std.Random.DefaultPrng;
const Allocator = std.mem.Allocator;

const Value = @import("../value.zig").Value;

/// Neuron represents a single neuron in a neural network.
pub const Neuron = @This();

w: []Value(f32),
b: *Value(f32),
non_linear: bool = true,
parameters: []*Value(f32),
allocator: Allocator,
arena: std.heap.ArenaAllocator,

/// Creates a neuron given the input values.
pub fn init(allocator: Allocator, num_input: usize, non_linear: bool) Allocator.Error!Neuron {
    const w = try allocator.alloc(Value(f32), num_input);
    const b = blk: {
        const bp = try allocator.create(Value(f32));
        bp.* = Value(f32).init(0.0);
        break :blk bp;
    };
    const parameters = try allocator.alloc(*Value(f32), num_input + 1); // +1 for bias

    var prng = Prng.init(@intCast(std.time.milliTimestamp()));
    for (w, 0..) |*weight, i| {
        weight.* = Value(f32).init(prng.random().float(f32) * 2.0 - 1.0); // [-1.0, 1.0)
        parameters[i] = weight;
    }
    parameters[parameters.len - 1] = b; // Last parameter is the bias

    return Neuron{
        .w = w,
        .b = b,
        .non_linear = non_linear,
        .parameters = parameters,
        .allocator = allocator,
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
}

/// Deinitializes the neuron and frees its resources.
pub fn deinit(self: Neuron) void {
    self.arena.deinit();
    self.allocator.free(self.parameters);
    self.allocator.destroy(self.b);
    self.allocator.free(self.w);
}

pub fn forward(self: *Neuron, x: []const *Value(f32)) Allocator.Error!*Value(f32) {
    const allocator = self.arena.allocator();
    const products = try allocator.alloc(Value(f32), self.w.len);
    // products = [w1 * x1, w2 * x2, ..., wn * xn]
    for (products, self.w, x) |*product, *w, xi| {
        product.* = w.mul(xi);
    }
    const sums = try allocator.alloc(Value(f32), self.w.len);
    // sum = w1 * x1 + w2 * x2 + ... + wn * xn
    sums[0] = products[0];
    for (products[1..], 1..) |*p, i| {
        sums[i] = sums[i - 1].add(p);
    }
    // z = sum + b
    const z = try allocator.create(Value(f32));
    z.* = sums[sums.len - 1].add(self.b);
    if (!self.non_linear) {
        return z;
    } else {
        const relu = try allocator.create(Value(f32));
        relu.* = z.relu();
        return relu;
    }
}

pub fn zeroGrad(self: *Neuron) void {
    for (self.parameters) |p| {
        p.grad = 0.0;
    }
}

const testing = @import("std").testing;

test init {
    const allocator = std.testing.allocator;
    for (0..5) |_| {
        const neuron = try Neuron.init(allocator, 3, true);
        defer neuron.deinit();

        for (neuron.w) |weight| {
            try testing.expect(weight.data >= -1.0 and weight.data < 1.0);
        }
        try testing.expectEqual(3, neuron.w.len);
        try testing.expectEqual(0.0, neuron.b.data);
        try testing.expectEqual(true, neuron.non_linear);

        try testing.expectEqual(4, neuron.parameters.len); // 3 weights + 1 bias
        try testing.expectEqual(&neuron.w[0], neuron.parameters[0]);
        try testing.expectEqual(&neuron.w[1], neuron.parameters[1]);
        try testing.expectEqual(&neuron.w[2], neuron.parameters[2]);
        try testing.expectEqual(neuron.b, neuron.parameters[3]);
    }
}

test forward {
    const allocator = std.testing.allocator;
    var neuron = try Neuron.init(allocator, 3, true);
    defer neuron.deinit();

    // w is randomly generated. Reset them to fixed value for easy testing.
    neuron.w[0] = Value(f32).init(-1.0);
    neuron.w[1] = Value(f32).init(-0.25);
    neuron.w[2] = Value(f32).init(0.75);

    var x1 = Value(f32).init(1.0);
    var x2 = Value(f32).init(2.0);
    var x3 = Value(f32).init(3.0);
    var x = [_]*Value(f32){
        &x1,
        &x2,
        &x3,
    };
    const output1 = try neuron.forward(x[0..]);
    try testing.expectEqual(0.75, output1.data);

    x3 = Value(f32).init(-1.0);
    const output2 = try neuron.forward(x[0..]);
    try testing.expectEqual(0.0, output2.data); // since relu is applied
}

test "foward without non_linear" {
    const allocator = std.testing.allocator;
    var neuron = try Neuron.init(allocator, 2, false);
    defer neuron.deinit();

    // w is randomly generated. Reset them to fixed value for easy testing.
    neuron.w[0] = Value(f32).init(-1.0);
    neuron.w[1] = Value(f32).init(-0.25);

    var x1 = Value(f32).init(1.0);
    var x2 = Value(f32).init(2.0);
    var x = [_]*Value(f32){
        &x1,
        &x2,
    };
    const output = try neuron.forward(x[0..]);
    try testing.expectEqual(-1.5, output.data);
}

test zeroGrad {
    const allocator = std.testing.allocator;
    var neuron = try Neuron.init(allocator, 3, true);
    defer neuron.deinit();

    // Set some gradients
    for (neuron.parameters) |p| {
        p.grad = 1.0;
    }

    // Zero the gradients
    neuron.zeroGrad();

    // Check that all gradients are zero
    for (neuron.parameters) |p| {
        try testing.expectEqual(0.0, p.grad);
    }
}
