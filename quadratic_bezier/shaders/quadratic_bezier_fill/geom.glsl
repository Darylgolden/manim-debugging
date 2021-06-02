#version 330

layout (triangles) in;
layout (triangle_strip, max_vertices = 5) out;

uniform float anti_alias_width;

// Needed for get_gl_Position
uniform vec2 frame_shape;
uniform float focal_distance;
uniform float is_fixed_in_frame;
// Needed for finalize_color
uniform vec3 light_source_position;
uniform float gloss;
uniform float shadow;

in vec3 bp[3];
in vec3 v_global_unit_normal[3];
in vec4 v_color[3];
in float v_vert_index[3];

out vec4 color;
out float fill_all;
out float uv_anti_alias_width;

out vec3 xyz_coords;
out float orientation;
// uv space is where b0 = (0, 0), b1 = (1, 0), and transform is orthogonal
out vec2 uv_coords;
out vec2 uv_b2;
out float bezier_degree;


// Analog of import for manim only
// #include ../include/quadratic_bezier_geometry_functions.glsl
// ------------------------------------------------------------

float cross2d(vec2 v, vec2 w){
    return v.x * w.y - w.x * v.y;
}


mat3 get_xy_to_uv(vec2 b0, vec2 b1){
    mat3 shift = mat3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        -b0.x, -b0.y, 1.0
    );

    float sf = length(b1 - b0);
    vec2 I = (b1 - b0) / sf;
    vec2 J = vec2(-I.y, I.x);
    mat3 rotate = mat3(
        I.x, J.x, 0.0,
        I.y, J.y, 0.0,
        0.0, 0.0, 1.0
    );
    return (1 / sf) * rotate * shift;
}


// Orthogonal matrix to convert to a uv space defined so that
// b0 goes to [0, 0] and b1 goes to [1, 0]
mat4 get_xyz_to_uv(vec3 b0, vec3 b1, vec3 unit_normal){
    mat4 shift = mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -b0.x, -b0.y, -b0.z, 1
    );

    float scale_factor = length(b1 - b0);
    vec3 I = (b1 - b0) / scale_factor;
    vec3 K = unit_normal;
    vec3 J = cross(K, I);
    // Transpose (hence inverse) of matrix taking
    // i-hat to I, k-hat to unit_normal, and j-hat to their cross
    mat4 rotate = mat4(
        I.x, J.x, K.x, 0.0,
        I.y, J.y, K.y, 0.0,
        I.z, J.z, K.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    return (1 / scale_factor) * rotate * shift;
}


// Returns 0 for null curve, 1 for linear, 2 for quadratic.
// Populates new_points with bezier control points for the curve,
// which for quadratics will be the same, but for linear and null
// might change.  The idea is to inform the caller of the degree,
// while also passing tangency information in the linear case.
// float get_reduced_control_points(vec3 b0, vec3 b1, vec3 b2, out vec3 new_points[3]){
float get_reduced_control_points(in vec3 points[3], out vec3 new_points[3]){
    float length_threshold = 1e-6;
    float angle_threshold = 5e-2;

    vec3 p0 = points[0];
    vec3 p1 = points[1];
    vec3 p2 = points[2];
    vec3 v01 = (p1 - p0);
    vec3 v12 = (p2 - p1);

    float dot_prod = clamp(dot(normalize(v01), normalize(v12)), -1, 1);
    bool aligned = acos(dot_prod) < angle_threshold;
    bool distinct_01 = length(v01) > length_threshold;  // v01 is considered nonzero
    bool distinct_12 = length(v12) > length_threshold;  // v12 is considered nonzero
    int n_uniques = int(distinct_01) + int(distinct_12);

    bool quadratic = (n_uniques == 2) && !aligned;
    bool linear = (n_uniques == 1) || ((n_uniques == 2) && aligned);
    bool constant = (n_uniques == 0);
    if(quadratic){
        new_points[0] = p0;
        new_points[1] = p1;
        new_points[2] = p2;
        return 2.0;
    }else if(linear){
        new_points[0] = p0;
        new_points[1] = (p0 + p2) / 2.0;
        new_points[2] = p2;
        return 1.0;
    }else{
        new_points[0] = p0;
        new_points[1] = p0;
        new_points[2] = p0;
        return 0.0;
    }
}

// ------------------------------------------------------------
// #include ../include/get_gl_Position.glsl
// ------------------------------------------------------------

const vec2 DEFAULT_FRAME_SHAPE = vec2(8.0 * 16.0 / 9.0, 8.0);

float perspective_scale_factor(float z, float focal_distance){
    return max(0.0, focal_distance / (focal_distance - z));
}

vec4 get_gl_Position(vec3 point){
    vec4 result = vec4(point, 1.0);
    if(!bool(is_fixed_in_frame)){
        result.x *= 2.0 / frame_shape.x;
        result.y *= 2.0 / frame_shape.y;
        float psf = perspective_scale_factor(result.z, focal_distance);
        if (psf > 0){
            result.xy *= psf;
            // TODO, what's the better way to do this?
            // This is to keep vertices too far out of frame from getting cut.
            result.z *= 0.01;
        }
    } else{
        result.x *= 2.0 / DEFAULT_FRAME_SHAPE.x;
        result.y *= 2.0 / DEFAULT_FRAME_SHAPE.y;
    }
    result.z *= -1;
    return result;
}

// ------------------------------------------------------------
// #include ../include/get_unit_normal.glsl
// ------------------------------------------------------------

vec3 get_unit_normal(in vec3[3] points){
    float tol = 1e-6;
    vec3 v1 = normalize(points[1] - points[0]);
    vec3 v2 = normalize(points[2] - points[0]);
    vec3 cp = cross(v1, v2);
    float cp_norm = length(cp);
    if(cp_norm < tol){
        // Three points form a line, so find a normal vector
        // to that line in the plane shared with the z-axis
        vec3 k_hat = vec3(0.0, 0.0, 1.0);
        vec3 new_cp = cross(cross(v2, k_hat), v2);
        float new_cp_norm = length(new_cp);
        if(new_cp_norm < tol){
            // We only come here if all three points line up
            // on the z-axis.
            return vec3(0.0, -1.0, 0.0);
            // return k_hat;
        }
        return new_cp / new_cp_norm;
    }
    return cp / cp_norm;
}

// ------------------------------------------------------------
// #include ../include/finalize_color.glsl
// ------------------------------------------------------------

vec3 float_to_color(float value, float min_val, float max_val, vec3[9] colormap_data){
    float alpha = clamp((value - min_val) / (max_val - min_val), 0.0, 1.0);
    int disc_alpha = min(int(alpha * 8), 7);
    return mix(
        colormap_data[disc_alpha],
        colormap_data[disc_alpha + 1],
        8.0 * alpha - disc_alpha
    );
}

vec4 add_light(vec4 color,
               vec3 point,
               vec3 unit_normal,
               vec3 light_coords,
               float gloss,
               float shadow){
    if(gloss == 0.0 && shadow == 0.0) return color;

    // TODO, do we actually want this?  It effectively treats surfaces as two-sided
    if(unit_normal.z < 0){
            unit_normal *= -1;
    }

    // TODO, read this in as a uniform?
    float camera_distance = 6;  
    // Assume everything has already been rotated such that camera is in the z-direction
    vec3 to_camera = vec3(0, 0, camera_distance) - point;
    vec3 to_light = light_coords - point;
    vec3 light_reflection = -to_light + 2 * unit_normal * dot(to_light, unit_normal);
    float dot_prod = dot(normalize(light_reflection), normalize(to_camera));
    float shine = gloss * exp(-3 * pow(1 - dot_prod, 2));
    float dp2 = dot(normalize(to_light), unit_normal);
    float darkening = mix(1, max(dp2, 0), shadow);
    return vec4(
            darkening * mix(color.rgb, vec3(1.0), shine),
            color.a
    );
}

vec4 finalize_color(vec4 color,
                    vec3 point,
                    vec3 unit_normal,
                    vec3 light_coords,
                    float gloss,
                    float shadow){
    ///// INSERT COLOR FUNCTION HERE /////
    // The line above may be replaced by arbitrary code snippets, as per
    // the method Mobject.set_color_by_code
    return add_light(color, point, unit_normal, light_coords, gloss, shadow);
}

// ------------------------------------------------------------

void emit_vertex_wrapper(vec3 point, int index){
    color = finalize_color(
        v_color[index],
        point,
        v_global_unit_normal[index],
        light_source_position,
        gloss,
        shadow
    );
    xyz_coords = point;
    gl_Position = get_gl_Position(xyz_coords);
    EmitVertex();
}


void emit_simple_triangle(){
    for(int i = 0; i < 3; i++){
        emit_vertex_wrapper(bp[i], i);
    }
    EndPrimitive();
}


void emit_pentagon(vec3[3] points, vec3 normal){
    vec3 p0 = points[0];
    vec3 p1 = points[1];
    vec3 p2 = points[2];
    // Tangent vectors
    vec3 t01 = normalize(p1 - p0);
    vec3 t12 = normalize(p2 - p1);
    // Vectors perpendicular to the curve in the plane of the curve pointing outside the curve
    vec3 p0_perp = cross(t01, normal);
    vec3 p2_perp = cross(t12, normal);

    bool fill_inside = orientation > 0;
    float aaw = anti_alias_width;
    vec3 corners[5];
    if(fill_inside){
        // Note, straight lines will also fall into this case, and since p0_perp and p2_perp
        // will point to the right of the curve, it's just what we want
        corners = vec3[5](
            p0 + aaw * p0_perp,
            p0,
            p1 + 0.5 * aaw * (p0_perp + p2_perp),
            p2,
            p2 + aaw * p2_perp
        );
    }else{
        corners = vec3[5](
            p0,
            p0 - aaw * p0_perp,
            p1,
            p2 - aaw * p2_perp,
            p2
        );
    }

    mat4 xyz_to_uv = get_xyz_to_uv(p0, p1, normal);
    uv_b2 = (xyz_to_uv * vec4(p2, 1)).xy;
    uv_anti_alias_width = anti_alias_width / length(p1 - p0);

    for(int i = 0; i < 5; i++){
        vec3 corner = corners[i];
        uv_coords = (xyz_to_uv * vec4(corner, 1)).xy;
        int j = int(sign(i - 1) + 1);  // Maps i = [0, 1, 2, 3, 4] onto j = [0, 0, 1, 2, 2]
        emit_vertex_wrapper(corner, j);
    }
    EndPrimitive();
}


void main(){
    // If vert indices are sequential, don't fill all
    fill_all = float(
        (v_vert_index[1] - v_vert_index[0]) != 1.0 ||
        (v_vert_index[2] - v_vert_index[1]) != 1.0
    );

    if(fill_all == 1.0){
        emit_simple_triangle();
        return;
    }

    vec3 new_bp[3];
    bezier_degree = get_reduced_control_points(vec3[3](bp[0], bp[1], bp[2]), new_bp);
    vec3 local_unit_normal = get_unit_normal(new_bp);
    orientation = sign(dot(v_global_unit_normal[0], local_unit_normal));

    if(bezier_degree >= 1){
        emit_pentagon(new_bp, local_unit_normal);
    }
    // Don't emit any vertices for bezier_degree 0
}

