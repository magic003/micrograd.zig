const std = @import("std");
const Allocator = std.mem.Allocator;

const Neuron = @import("neuron.zig").Neuron;
const Value = @import("../value.zig").Value;

/// Layer represents a layer in a neural network, containing multiple neurons.
pub const Layer = struct {
    neurons: []Neuron,
    outputs: []Value(f32),
    allocator: Allocator,

    /// Creates a new layer with the specified number of inputs and outputs.
    pub fn init(
        allocator: Allocator,
        num_inputs: usize,
        num_outputs: usize,
        non_linear: bool,
    ) Allocator.Error!Layer {
        const neurons = try allocator.alloc(Neuron, num_outputs);
        for (neurons) |*neuron| {
            neuron.* = try Neuron.init(allocator, num_inputs, non_linear);
        }
        return Layer{
            .neurons = neurons,
            .outputs = try allocator.alloc(Value(f32), num_outputs),
            .allocator = allocator,
        };
    }

    /// Deinitializes the layer and frees its resources.
    pub fn deinit(self: Layer) void {
        for (self.neurons) |neuron| {
            neuron.deinit();
        }
        self.allocator.free(self.outputs);
        self.allocator.free(self.neurons);
    }

    pub fn forward(self: *Layer, x: []*Value(f32)) Allocator.Error![]Value(f32) {
        for (self.neurons, self.outputs) |*neuron, *output| {
            output.* = try neuron.forward(x);
        }
        return self.outputs;
    }

    const testing = @import("std").testing;

    test init {
        const allocator = std.testing.allocator;
        const layer = try Layer.init(allocator, 3, 4, true);
        defer layer.deinit();

        try testing.expectEqual(4, layer.neurons.len);
        try testing.expectEqual(4, layer.outputs.len);
        for (layer.neurons) |neuron| {
            try testing.expectEqual(3, neuron.w.len);
            try testing.expectEqual(true, neuron.non_linear);
        }
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
};
