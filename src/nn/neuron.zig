const std = @import("std");
const Prng = std.Random.DefaultPrng;
const Allocator = std.mem.Allocator;
const Value = @import("../value.zig").Value;

/// Neuron represents a single neuron in a neural network.
pub const Neuron = struct {
    w: []Value(f32),
    b: Value(f32),
    non_linear: bool = true,

    /// Creates a neuron given the input values.
    pub fn init(allocator: Allocator, num_inputs: usize, non_linear: bool) Allocator.Error!Neuron {
        var prng = Prng.init(@intCast(std.time.milliTimestamp()));
        const w = try allocator.alloc(Value(f32), num_inputs);
        for (w) |*weight| {
            weight.* = Value(f32).init(prng.random().float(f32) * 2.0 - 1.0); // [-1.0, 1.0)
        }
        return Neuron{
            .w = w,
            .b = Value(f32).init(0.0),
            .non_linear = non_linear,
        };
    }

    /// Deinitializes the neuron and frees its resources.
    pub fn deinit(self: Neuron, allocator: Allocator) void {
        allocator.free(self.w);
    }

    const testing = @import("std").testing;

    test init {
        const allocator = std.testing.allocator;
        for (0..5) |_| {
            const neuron = try Neuron.init(allocator, 3, true);
            defer neuron.deinit(allocator);

            for (neuron.w) |weight| {
                try testing.expect(weight.data >= -1.0 and weight.data < 1.0);
            }
            try testing.expectEqual(3, neuron.w.len);
            try testing.expectEqual(0.0, neuron.b.data);
            try testing.expectEqual(true, neuron.non_linear);
        }
    }
};
