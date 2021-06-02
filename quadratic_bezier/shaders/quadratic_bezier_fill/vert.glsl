#version 330

// #include ../include/camera_uniform_declarations.glsl
// -----------------------------------------------------

uniform vec2 frame_shape;
uniform float anti_alias_width;
uniform vec3 camera_center;
uniform mat3 camera_rotation;
uniform float is_fixed_in_frame;
uniform float focal_distance;

// -----------------------------------------------------

in vec3 point;
in vec3 unit_normal;
in vec4 color;
in float vert_index;

out vec3 bp;  // Bezier control point
out vec3 v_global_unit_normal;
out vec4 v_color;
out float v_vert_index;

// Analog of import for manim only
// #include ../include/position_point_into_frame.glsl
// --------------------------------------------------

vec3 rotate_point_into_frame(vec3 point){
    if(bool(is_fixed_in_frame)){
        return point;
    }
    return camera_rotation * point;
}


vec3 position_point_into_frame(vec3 point){
    if(bool(is_fixed_in_frame)){
        return point;
    }
    return rotate_point_into_frame(point - camera_center);
}

// --------------------------------------------------

void main(){
    bp = position_point_into_frame(point.xyz);
    v_global_unit_normal = rotate_point_into_frame(unit_normal.xyz);
    v_color = color;
    v_vert_index = vert_index;
}
