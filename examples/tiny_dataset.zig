const std = @import("std");
const mg = @import("micrograd");

const Vf32 = mg.Value(f32);
/// A simple example of a multi-layer perceptron (MLP) trained on a tiny dataset.
/// Inspired by the [tiny dataset](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6664s) part in
/// Andrej Karpathy's micrograd video.
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const IN = 3;
    const layers = [_]usize{ 4, 3, 1 };
    const num_parameters = blk: {
        comptime var sum = 0;
        comptime var input = IN;
        inline for (layers) |num_output| {
            sum += (input + 1) * num_output; // weights + biases
            input = num_output;
        }
        break :blk sum;
    };
    var mlp = try mg.MLP.init(gpa.allocator(), IN, &layers);
    defer mlp.deinit();

    var arena_allocator = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena_allocator.deinit();

    var xs = [_][3]mg.Value(f32){
        .{ Vf32.init(2.0), Vf32.init(3.0), Vf32.init(-1.1) },
        .{ Vf32.init(3.5), Vf32.init(-1.0), Vf32.init(0.5) },
        .{ Vf32.init(0.7), Vf32.init(1.3), Vf32.init(1.9) },
        .{ Vf32.init(1.1), Vf32.init(1.8), Vf32.init(-1.3) },
    };
    const ys_f32 = [_]f32{
        1.0,
        2.2,
        3.5,
        0.0,
    };

    const learning_rate = 0.01;
    for (0..5) |epoch| {
        var ys: [ys_f32.len]*Vf32 = undefined;
        for (0..ys_f32.len) |i| {
            ys[i] = try arena_allocator.allocator().create(Vf32);
            ys[i].* = Vf32.init(ys_f32[i]);
        }
        std.debug.print("Epoch: {d}\n", .{epoch});
        printValues("Parameters", num_parameters, mlp.parameters);
        var ypred: [ys.len]*Vf32 = undefined;
        for (0..ypred.len) |i| {
            // Forward pass
            const x0 = &xs[i][0];
            const x1 = &xs[i][1];
            const x2 = &xs[i][2];
            printValues("Input", 3, xs[i]);
            const output = try mlp.forward(&.{ x0, x1, x2 });
            ypred[i] = output[0];
        }
        printValues("Predicted", ypred.len, ypred);
        printValues("Expected", ys.len, ys);
        var squared_error: [ys.len]*Vf32 = undefined;
        for (0..ys.len) |i| {
            var diff = try arena_allocator.allocator().create(Vf32);
            diff.* = ys[i].sub(ypred[i]);
            const squared = try arena_allocator.allocator().create(Vf32);
            squared.* = diff.pow(2.0);
            squared_error[i] = squared;
        }
        printValues("Squared Error: ", squared_error.len, squared_error);

        var loss = squared_error[0];
        for (squared_error[1..]) |se| {
            const sum = try arena_allocator.allocator().create(Vf32);
            sum.* = loss.add(se);
            loss = sum;
        }

        mlp.zeroGrad();
        try loss.backward();
        std.debug.print("Loss: {d:.8}, grad: {d:.8}\n", .{ loss.data, loss.grad });
        for (mlp.parameters) |param| {
            // Update parameters
            param.data -= param.grad * learning_rate;
        }
        std.debug.print("============================\n", .{});
    }
}

fn printValues(comptime msg: []const u8, comptime N: usize, values: anytype) void {
    var data: [N]f32 = undefined;
    var grad: [N]f32 = undefined;
    for (values, 0..) |v, i| {
        data[i] = v.data;
        grad[i] = v.grad;
    }
    std.debug.print("{s}: data={d:.8}, grad={d:.8}\n", .{ msg, data, grad });
}
