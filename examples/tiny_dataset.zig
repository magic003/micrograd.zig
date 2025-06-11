const std = @import("std");
const mg = @import("micrograd");

const Vf32 = mg.Value(f32);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var mlp = try mg.MLP.init(gpa.allocator(), 3, &.{ 4, 4, 1 });
    defer mlp.deinit();

    const xs = [4][3]mg.Value(f32){
        .{ Vf32.init(2.0), Vf32.init(3.0), Vf32.init(-1.0) },
        .{ Vf32.init(3.0), Vf32.init(-1.0), Vf32.init(0.5) },
        .{ Vf32.init(0.5), Vf32.init(1.0), Vf32.init(1.0) },
        .{ Vf32.init(1.0), Vf32.init(1.0), Vf32.init(-1.0) },
    };
    const ys = [_]mg.Value(f32){
        Vf32.init(1.0),
        Vf32.init(-1.0),
        Vf32.init(-1.0),
        Vf32.init(1.0),
    };

    const learning_rate = 0.01;
    for (0..6) |_| {
        var ypred: [4]Vf32 = undefined;
        for (&ypred, 0..) |*yp, i| {
            // Forward pass
            var x0 = xs[i][0];
            var x1 = xs[i][1];
            var x2 = xs[i][2];
            const output = try mlp.forward(&.{ &x0, &x1, &x2 });
            yp.* = output[0];
        }
        printValues("Predicted", ypred.len, ypred);
        var squared_error: [4]Vf32 = undefined;
        for (&squared_error, 0..) |*se, i| {
            var y = ys[i];
            var diff = y.sub(&ypred[i]);
            se.* = diff.pow(2.0);
        }
        printValues("Squared Error: ", squared_error.len, squared_error);

        var loss = squared_error[0];
        for (squared_error[1..]) |*se| {
            loss = loss.add(se);
        }
        std.debug.print("Loss: {d:.8}\n", .{loss.data});

        mlp.zeroGrad();
        try loss.backward();
        for (mlp.parameters) |param| {
            // Update parameters
            param.data -= param.grad * learning_rate;
        }
        std.debug.print("============================\n", .{});
    }
}

fn printValues(comptime msg: []const u8, comptime N: usize, values: [N]Vf32) void {
    const vs: [N]f32 = blk: {
        var arr: [N]f32 = undefined;
        for (values, 0..) |v, i| {
            arr[i] = v.data;
        }
        break :blk arr;
    };
    std.debug.print("{s}: {d:.8}\n", .{ msg, vs });
}
