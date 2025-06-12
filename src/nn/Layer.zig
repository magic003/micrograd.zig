const std = @import("std");
const Allocator = std.mem.Allocator;

const Neuron = @import("neuron.zig");
const Value = @import("../value.zig").Value;

/// Layer represents a layer in a neural network, containing multiple neurons.
pub const Layer = @This();

neurons: []Neuron,
parameters: []*Value(f32),
allocator: Allocator,
arena: std.heap.ArenaAllocator,

/// Creates a new layer with the specified number of inputs and outputs.
pub fn init(
    allocator: Allocator,
    num_input: usize,
    num_output: usize,
    non_linear: bool,
) Allocator.Error!Layer {
    const neurons = try allocator.alloc(Neuron, num_output);
    const parameters = try allocator.alloc(*Value(f32), (num_input + 1) * num_output);
    for (neurons, 0..) |*neuron, i| {
        neuron.* = try Neuron.init(allocator, num_input, non_linear);
        for (neuron.parameters, 0..) |param, j| {
            parameters[i * (num_input + 1) + j] = param;
        }
    }
    return Layer{
        .neurons = neurons,
        .parameters = parameters,
        .allocator = allocator,
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
}

/// Deinitializes the layer and frees its resources.
pub fn deinit(self: Layer) void {
    self.arena.deinit();
    for (self.neurons) |neuron| {
        neuron.deinit();
    }
    self.allocator.free(self.parameters);
    self.allocator.free(self.neurons);
}

pub fn forward(self: *Layer, x: []const *Value(f32)) Allocator.Error![]*Value(f32) {
    const outputs = try self.arena.allocator().alloc(*Value(f32), self.neurons.len);
    for (self.neurons, outputs) |*neuron, *output| {
        output.* = try neuron.forward(x);
    }
    return outputs;
}

pub fn zeroGrad(self: *Layer) void {
    for (self.neurons) |*neuron| {
        neuron.zeroGrad();
    }
}

const testing = @import("std").testing;

test init {
    const allocator = std.testing.allocator;
    const layer = try Layer.init(allocator, 3, 4, true);
    defer layer.deinit();

    try testing.expectEqual(4, layer.neurons.len);
    for (layer.neurons) |neuron| {
        try testing.expectEqual(3, neuron.w.len);
        try testing.expectEqual(true, neuron.non_linear);
    }

    try testing.expectEqual(16, layer.parameters.len); // 3 weights + 1 bias for each neuron
    try testing.expectEqual(&layer.neurons[0].w[0], layer.parameters[0]);
    try testing.expectEqual(&layer.neurons[0].w[1], layer.parameters[1]);
    try testing.expectEqual(&layer.neurons[0].w[2], layer.parameters[2]);
    try testing.expectEqual(layer.neurons[0].b, layer.parameters[3]);
}

test forward {
    const allocator = std.testing.allocator;
    var layer = try Layer.init(allocator, 3, 2, true);
    defer layer.deinit();

    // w is randomly generated. Reset them to fixed value for easy testing.
    layer.neurons[0].w[0] = Value(f32).init(-1.0);
    layer.neurons[0].w[1] = Value(f32).init(-0.25);
    layer.neurons[0].w[2] = Value(f32).init(0.75);

    layer.neurons[1].w[0] = Value(f32).init(0.5);
    layer.neurons[1].w[1] = Value(f32).init(-0.5);
    layer.neurons[1].w[2] = Value(f32).init(-1.0);

    var x1 = Value(f32).init(1.0);
    var x2 = Value(f32).init(2.0);
    var x3 = Value(f32).init(3.0);
    var x = [_]*Value(f32){
        &x1,
        &x2,
        &x3,
    };
    const outputs = try layer.forward(x[0..]);

    try testing.expectEqual(2, outputs.len);
    try testing.expectEqual(0.75, outputs[0].data); // Neuron 1 output
    try testing.expectEqual(0.0, outputs[1].data); // Neuron 2 output
}

test zeroGrad {
    const allocator = std.testing.allocator;
    var layer = try Layer.init(allocator, 3, 2, true);
    defer layer.deinit();

    // Set some gradients
    for (layer.parameters) |p| {
        p.grad = 1.0;
    }

    // Zero gradients
    layer.zeroGrad();

    // Check if gradients are zeroed
    for (layer.neurons) |neuron| {
        for (neuron.parameters) |param| {
            try testing.expectEqual(0.0, param.grad);
        }
    }
}
