const Data = @import("data.zig").Data;

/// A value stores a scalar data and its gradient.
pub const Value = @This();

/// The scalar data.
data: Data,

/// Creates a value with the given type and value.
pub fn init(comptime T: type, value: T) Value {
    return Value{
        .data = Data.init(T, value),
    };
}

const testing = @import("std").testing;
test init {
    const value_f32 = Value.init(f32, 10.0);
    const value_i32 = Value.init(i32, 10);

    try testing.expectEqual(10.0, value_f32.data.f32);
    try testing.expectEqual(10, value_i32.data.i32);
}
