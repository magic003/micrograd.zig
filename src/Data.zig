/// The data saved in a value.
pub const Data = union(enum) {
    f32: f32,
    i32: i32,

    /// Creates a data with the given type and value.
    pub fn init(comptime T: type, value: T) Data {
        return switch (T) {
            f32 => Data{ .f32 = value },
            i32 => Data{ .i32 = value },
            else => @compileError("Unsupported data type: " ++ @typeName(T)),
        };
    }

    const testing = @import("std").testing;
    test init {
        const data_f32 = Data.init(f32, 10.0);
        const data_i32 = Data.init(i32, 10);

        try testing.expectEqual(10.0, data_f32.f32);
        try testing.expectEqual(10, data_i32.i32);
    }
};
