const std = @import("std");
const Allocator = std.mem.Allocator;

const Neuron = @import("neuron.zig").Neuron;

/// Layer represents a layer in a neural network, containing multiple neurons.
pub const Layer = struct {
    neurons: []Neuron,
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
        return Layer{ .neurons = neurons, .allocator = allocator };
    }

    /// Deinitializes the layer and frees its resources.
    pub fn deinit(self: Layer) void {
        for (self.neurons) |neuron| {
            neuron.deinit();
        }
        self.allocator.free(self.neurons);
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
    }
};
