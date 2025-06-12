const std = @import("std");
const Allocator = std.mem.Allocator;

const Layer = @import("layer.zig");
const Value = @import("../value.zig").Value;

/// MLP represents a Multi-Layer Perceptron (MLP) neural network.
pub const MLP = @This();

layers: []Layer,
parameters: []*Value(f32),
allocator: std.mem.Allocator,

pub fn init(
    allocator: Allocator,
    num_input: usize,
    num_outputs: []const usize,
) Allocator.Error!MLP {
    const layers = try allocator.alloc(Layer, num_outputs.len);
    var num_parameter: usize = 0;
    var input = num_input; // Start with the initial input size
    for (num_outputs, layers) |num_output, *layer| {
        layer.* = try Layer.init(allocator, input, num_output, true);
        input = num_output; // Update input size for the next layer
        num_parameter += layer.parameters.len;
    }

    const parameters = blk: {
        const result = try allocator.alloc(*Value(f32), num_parameter);
        var index: usize = 0;
        for (layers) |layer| {
            for (layer.parameters) |param| {
                result[index] = param;
                index += 1;
            }
        }
        break :blk result;
    };
    return MLP{
        .layers = layers,
        .parameters = parameters,
        .allocator = allocator,
    };
}

pub fn deinit(self: MLP) void {
    for (self.layers) |layer| {
        layer.deinit();
    }
    self.allocator.free(self.parameters);
    self.allocator.free(self.layers);
}

pub fn forward(self: *MLP, x: []const *Value(f32)) Allocator.Error![]*Value(f32) {
    var input = x;
    var output: []*Value(f32) = undefined;
    for (self.layers) |*layer| {
        output = try layer.forward(input);
        input = output;
    }

    return output;
}

pub fn zeroGrad(self: *MLP) void {
    for (self.layers) |*layer| {
        layer.zeroGrad();
    }
}

const testing = @import("std").testing;

test init {
    const allocator = std.testing.allocator;
    var num_outputs = [_]usize{ 4, 5, 2 };
    const mlp = try MLP.init(allocator, 3, num_outputs[0..]);
    defer mlp.deinit();

    try testing.expectEqual(3, mlp.layers.len);
    try testing.expectEqual(3, mlp.layers[0].neurons[0].w.len);
    try testing.expectEqual(4, mlp.layers[0].neurons.len);
    try testing.expectEqual(mlp.layers[0].neurons.len, mlp.layers[1].neurons[0].w.len);
    try testing.expectEqual(5, mlp.layers[1].neurons.len);
    try testing.expectEqual(mlp.layers[1].neurons.len, mlp.layers[2].neurons[0].w.len);
    try testing.expectEqual(2, mlp.layers[2].neurons.len);

    try testing.expectEqual(53, mlp.parameters.len);
}

test forward {
    const allocator = std.testing.allocator;
    var num_outputs = [_]usize{ 3, 2 };
    var mlp = try MLP.init(allocator, 2, num_outputs[0..]);
    defer mlp.deinit();

    // w is randomly generated. Reset them to fixed value for easy testing.
    // layer 1
    mlp.layers[0].neurons[0].w[0] = Value(f32).init(-1.0);
    mlp.layers[0].neurons[0].w[1] = Value(f32).init(-0.25);
    mlp.layers[0].neurons[1].w[0] = Value(f32).init(0.5);
    mlp.layers[0].neurons[1].w[1] = Value(f32).init(-0.5);
    mlp.layers[0].neurons[2].w[0] = Value(f32).init(0.75);
    mlp.layers[0].neurons[2].w[1] = Value(f32).init(0.25);

    // layer 2
    mlp.layers[1].neurons[0].w[0] = Value(f32).init(0.1);
    mlp.layers[1].neurons[0].w[1] = Value(f32).init(-0.2);
    mlp.layers[1].neurons[0].w[2] = Value(f32).init(0.3);
    mlp.layers[1].neurons[1].w[0] = Value(f32).init(-0.4);
    mlp.layers[1].neurons[1].w[1] = Value(f32).init(0.5);
    mlp.layers[1].neurons[1].w[2] = Value(f32).init(-0.6);

    var x1 = Value(f32).init(1.0);
    var x2 = Value(f32).init(2.0);
    var x = [_]*Value(f32){
        &x1,
        &x2,
    };

    const outputs = try mlp.forward(x[0..]);
    try testing.expectEqual(2, outputs.len);
    try testing.expectEqual(0.375, outputs[0].data);
    try testing.expectEqual(0.0, outputs[1].data);
}

test zeroGrad {
    const allocator = std.testing.allocator;
    var num_outputs = [_]usize{ 3, 2 };
    var mlp = try MLP.init(allocator, 2, num_outputs[0..]);
    defer mlp.deinit();

    // Set some gradients
    for (mlp.parameters) |p| {
        p.grad = 1.0;
    }

    mlp.zeroGrad();

    for (mlp.layers) |layer| {
        for (layer.neurons) |neuron| {
            for (neuron.parameters) |param| {
                try testing.expectEqual(0.0, param.grad);
            }
        }
    }
}
