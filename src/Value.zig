const Data = @import("data.zig").Data;

/// A value stores a scalar data and its gradient.
pub const Value = @This();

/// The type of the value.
T: type,
/// The scalar data.
data: Data,

/// Creates a value with the given type and value.
pub fn init(comptime T: type, value: T) Value {
    return switch (T) {
        f32 => .{
            .data = Data{ .f32 = value },
            .T = T,
        },
        i32 => .{
            .data = Data{ .i32 = value },
            .T = T,
        },
        else => @compileError("Unsupported data type: " ++ @typeName(T)),
    };
}

pub fn add(self: Value, other: Value) Value {
    if (self.T != other.T) {
        @compileError("Cannot add values of different types: " ++ @typeName(self.T) ++ " and " ++ @typeName(other.T));
    }
    return switch (self.T) {
        f32 => .{
            .data = Data{ .f32 = self.data.f32 + other.data.f32 },
            .T = self.T,
        },
        i32 => .{
            .data = Data{ .i32 = self.data.i32 + other.data.i32 },
            .T = self.T,
        },
        else => unreachable,
    };
}

const testing = @import("std").testing;

test init {
    const value_f32 = Value.init(f32, 10.0);
    const value_i32 = Value.init(i32, 10);

    try testing.expectEqual(10.0, value_f32.data.f32);
    try testing.expectEqual(f32, value_f32.T);
    try testing.expectEqual(10, value_i32.data.i32);
    try testing.expectEqual(i32, value_i32.T);
}

test add {
    const value_f32_1 = Value.init(f32, 10.0);
    const value_f32_2 = Value.init(f32, 5.0);
    const result_f32 = value_f32_1.add(value_f32_2);
    try testing.expectEqual(15.0, result_f32.data.f32);
    try testing.expectEqual(f32, result_f32.T);

    const value_i32_1 = Value.init(i32, 10);
    const value_i32_2 = Value.init(i32, 5);
    const result_i32 = value_i32_1.add(value_i32_2);
    try testing.expectEqual(15, result_i32.data.i32);
    try testing.expectEqual(i32, result_i32.T);
}
