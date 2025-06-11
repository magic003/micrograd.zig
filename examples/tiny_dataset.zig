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
    // const ys = [_]f32{ 1.0, -1.0, -1.0, 1.0 };

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
}

fn printValues(comptime msg: []const u8, comptime N: usize, values: [N]Vf32) void {
    const vs: [N]f32 = blk: {
        var arr: [N]f32 = undefined;
        for (values, 0..) |v, i| {
            arr[i] = v.data;
        }
        break :blk arr;
    };
    std.debug.print("{s}: {d:.10}\n", .{ msg, vs });
}
