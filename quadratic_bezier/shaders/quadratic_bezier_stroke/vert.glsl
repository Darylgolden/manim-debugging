#version 330

// #include ../include/camera_uniform_declarations.glsl
// ----------
uniform vec2 frame_shape;
uniform float anti_alias_width;
uniform vec3 camera_center;
uniform mat3 camera_rotation;
uniform float is_fixed_in_frame;
uniform float focal_distance;
// ----------

in vec3 point;
in vec3 prev_point;
in vec3 next_point;
in vec3 unit_normal;

in float stroke_width;
in vec4 color;

// Bezier control point
out vec3 bp;
out vec3 prev_bp;
out vec3 next_bp;
out vec3 v_global_unit_normal;

out float v_stroke_width;
out vec4 v_color;

const float STROKE_WIDTH_CONVERSION = 0.01;

// #include ../include/position_point_into_frame.glsl
// ---------
// Assumes the following uniforms exist in the surrounding context:
// uniform float is_fixed_in_frame;
// uniform vec3 camera_center;
// uniform mat3 camera_rotation;

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

// ---------

void main(){
    bp = position_point_into_frame(point);
    prev_bp = position_point_into_frame(prev_point);
    next_bp = position_point_into_frame(next_point);
    v_global_unit_normal = rotate_point_into_frame(unit_normal);

    v_stroke_width = STROKE_WIDTH_CONVERSION * stroke_width;
    v_color = color;
}
