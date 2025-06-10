const std = @import("std");
const Allocator = std.mem.Allocator;

const Layer = @import("layer.zig").Layer;

/// MLP represents a Multi-Layer Perceptron (MLP) neural network.
pub const MLP = struct {
    layers: []Layer,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: Allocator,
        num_input: usize,
        num_outputs: []usize,
    ) Allocator.Error!MLP {
        const layers = try allocator.alloc(Layer, num_outputs.len);
        var input = num_input; // Start with the initial input size
        for (num_outputs, layers) |num_output, *layer| {
            layer.* = try Layer.init(allocator, input, num_output, true);
            input = num_output; // Update input size for the next layer
        }
        return MLP{
            .layers = layers,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: MLP) void {
        for (self.layers) |layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
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
    }
};
