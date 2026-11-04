// Garvis plasma orb -- the approved design (2026-10-31; prototype and previews in
// prototypes/plasma_orb/). DO NOT change how it looks without the owner's approval.
//
// ONE source for two apps: Garvis Desktop runs this file as GLSL (app/ui/plasma.py)
// and Garvis Mobile ships a byte-identical copy (android/app/src/main/assets/
// plasma_orb.frag, guarded by tests/test_orb_shader_parity.py) that it runs as
// Android AGSL, replacing only the entry point below the marker line at the end.
//
// The whole orb is one pass that writes premultiplied light (rgb <= alpha), so it
// composites onto the dark UI as glow. GLSL subset shared by desktop GL 2.1
// (#version 120), GLSL ES 1.00 and AGSL: no textures, constant loop bounds,
// float maths only. The host prepends the #version / precision header.
//
// Layers, after docs/orb-reference/:
//   Reference 1 (primary)  -- an irregular, crumpled boundary made of several
//     bright overlapping folds; a translucent blue body full of filaments and
//     straight "shard" cracks at different depths; the hottest light in the
//     upper centre; sparks streaming off it.
//   Reference 2 (support)  -- a white-hot core with a ragged star of rays;
//     wavy, branching tendrils radiating into deep blue darkness.
// Colour: energy is accumulated as one HDR value and mapped through
// 1 - exp(-E * tint), with tint = Electric Blue #1677D2. Faint light is
// exactly Electric Blue, stronger light saturates blue first and turns cyan,
// the strongest turns near-white; Midnight Blue #102B4C is the haze around it.

uniform vec2 u_res;       // target size in pixels
uniform float u_time;     // seconds (sparks, slow wobble)
uniform float u_drift;    // plasma flow distance (advances with the state's flow)
uniform float u_swirl;    // radians: extra rotation while thinking
uniform float u_glow;     // 0..1 rim, tendrils, sparks, haze
uniform float u_heat;     // 0..1 core
uniform float u_plasma;   // 0..1 inner filaments
uniform float u_swell;    // radius multiplier (breathing, voice)
uniform float u_alert;    // 0..1 shift toward red
uniform float u_standby;  // 0..1 dim and calm
uniform float u_detail;   // 1 = full detail, 0 = compact (small sizes)

const float TAU = 6.28318530718;

vec3 hash33(vec3 p) {
    // "Hash without sine" (after Dave Hoskins): stable on GPUs and in float32.
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return -1.0 + 2.0 * fract((p.xxy + p.yxx) * p.zyx);
}

// Gradient noise in about [-0.8, 0.8], quintic fade.
float gnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = dot(hash33(i), f);
    float n100 = dot(hash33(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0));
    float n010 = dot(hash33(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0));
    float n110 = dot(hash33(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0));
    float n001 = dot(hash33(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0));
    float n101 = dot(hash33(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0));
    float n011 = dot(hash33(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0));
    float n111 = dot(hash33(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

// A thin bright line where the noise crosses zero: the filament primitive.
float ridge(float n, float sharpness) {
    return pow(1.0 - min(abs(n) * 1.6, 1.0), sharpness);
}

// Distance between the nearest two cell points (F2 - F1): small along cell
// borders, which read as straight, angular cracks -- the "shards" in Reference 1.
float cellEdge(vec3 x) {
    vec3 i = floor(x);
    vec3 f = x - i;
    float f1 = 8.0;
    float f2 = 8.0;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int xx = -1; xx <= 1; xx++) {
                vec3 g = vec3(float(xx), float(y), float(z));
                vec3 o = g + 0.5 + 0.45 * hash33(i + g) - f;
                float d = dot(o, o);
                if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
            }
        }
    }
    return sqrt(f2) - sqrt(f1);
}

vec3 rotY(vec3 q, float a) {
    float c = cos(a);
    float s = sin(a);
    return vec3(c * q.x + s * q.z, q.y, -s * q.x + c * q.z);
}

vec3 rotX(vec3 q, float a) {
    float c = cos(a);
    float s = sin(a);
    return vec3(q.x, c * q.y - s * q.z, s * q.y + c * q.z);
}

// fragCoord: the pixel centre, origin at the BOTTOM-left (+y up), like GLSL's fragment coordinate.
vec4 plasma(vec2 fragCoord) {
    // Orb space: the frame's shorter half-side is 1.0; +y is up.
    vec2 p = (fragCoord - 0.5 * u_res) / (0.5 * min(u_res.x, u_res.y));
    float r = length(p);
    vec2 dir = p / max(r, 1e-4);
    float ang = atan(p.y, p.x);
    float t = u_time;
    float full = step(0.5, u_detail);

    // Travel through noise space on a huge circle: bounded coordinates (no
    // float precision loss after hours), seamless, and never visibly repeating.
    float d = u_drift * 2.4;
    vec3 flow = 60.0 * vec3(cos(d / 60.0), sin(d / 60.0), 0.0);

    float R = 0.47 * u_swell;

    // 1. Irregular, lumpy silhouette (low + mid + fine frequencies).
    vec3 bq = vec3(dir * 1.0, 0.0) + flow * 0.30;
    float bump = 0.24 * gnoise(bq) + 0.075 * gnoise(vec3(dir * 2.9, 1.7) + flow * 0.55)
               + 0.022 * gnoise(vec3(dir * 8.0, 3.1) + flow * 1.1);
    float Rb = R * (1.0 + bump);
    float inside = smoothstep(Rb + 0.010, Rb - 0.040, r);

    // 2. The rim: several crumpled folds overlapping near the boundary, each
    //    broken into brighter and dimmer stretches.
    float rim = 0.0;
    for (int k = 0; k < 5; k++) {
        float fk = float(k);
        if (fk > 2.0 + 2.0 * full) break;
        vec3 q = vec3(dir * (1.4 + 0.55 * fk), fk * 7.31) + flow * (0.34 + 0.12 * fk);
        float rk = Rb * (1.0 + 0.14 * gnoise(q) + 0.05 * gnoise(q * 2.7 + 11.0) + 0.015 * gnoise(q * 8.0 + 4.0)
                       - 0.05 * fk);
        float w = 0.0045 + 0.006 * fk;
        float line = w / (abs(r - rk) + w);
        line *= line;
        float broken = 0.15 + 0.85 * ridge(gnoise(vec3(dir * (5.0 + fk), fk * 3.0) + flow * 1.3), 2.0);
        rim += line * broken * (1.0 - 0.13 * fk);
    }

    // 3. The body: filaments on nested shells, seen from the front (bright)
    //    and through the plasma at the back (dim) -- real parallax and depth.
    float fil = 0.0;
    float shards = 0.0;
    float folds = 0.0;
    for (int s = 0; s < 3; s++) {
        float fs = float(s);
        if (fs > 1.0 + full) break;
        float rs = Rb * (1.0 - 0.2 * fs);
        if (r < rs) {
            float z = sqrt(rs * rs - r * r);
            for (int h = 0; h < 2; h++) {
                float side = h == 0 ? 1.0 : -1.0;
                vec3 q = vec3(p, side * z) / rs;
                q = rotY(q, u_swirl * (1.0 + 0.6 * fs) + 0.09 * t * (1.0 + fs) * (0.5 - fs * 0.4));
                q = rotX(q, 0.35 + 0.15 * fs);
                vec3 wq = q * 1.6 + flow * (0.18 + 0.05 * fs) + fs * 9.7;
                vec3 warp = vec3(gnoise(wq), gnoise(wq + 17.3), gnoise(wq + 31.9));
                vec3 fq = q * (2.4 + 0.8 * fs) + warp * 1.1 + flow * (0.22 + 0.06 * fs);
                float f = 1.4 * ridge(gnoise(fq), 16.0) + 0.5 * ridge(gnoise(fq * 2.17 + 5.3), 22.0);
                // Light piles up where we look through more plasma (toward the shell's edge).
                float depth = 0.65 + 0.55 * (1.0 - z / rs);
                float facing = side > 0.0 ? 1.0 : 0.42;
                fil += f * depth * facing * (1.0 - 0.28 * fs);
                if (s == 0) {
                    // Large, crumpled sheets folding through the plasma (Reference 1).
                    float sheet = ridge(gnoise(q * 1.35 + warp * 0.5 + flow * 0.12 + 40.0), 7.0);
                    folds += sheet * facing * (0.55 + 0.45 * (1.0 - z / rs));
                }
                if (s == 0 && h == 0 && full > 0.5) {
                    float e = cellEdge(q * 1.9 + warp * 0.30 + flow * 0.08);
                    float crack = 1.0 - smoothstep(0.0, 0.028, e);
                    // Only broken stretches of the cell borders show: straight shards, never a closed tiling.
                    crack *= smoothstep(-0.12, 0.22, gnoise(q * 2.2 - flow * 0.2 + 3.0))
                           * smoothstep(-0.18, 0.15, gnoise(q * 7.5 + flow * 0.3 + 9.0));
                    shards += crack;
                }
            }
        }
    }
    fil *= inside;
    shards *= inside;
    folds *= inside;

    // 4. The core: white-hot, a little above centre, with a ragged star of rays.
    vec2 c0 = vec2(0.02, 0.07) * R / 0.47;
    vec2 pc = p - c0;
    float rc = length(pc);
    float core = exp(-rc * rc / (0.045 * R * R)) + 0.55 * exp(-rc / (0.24 * R));
    float ca = atan(pc.y, pc.x);
    float rays = ridge(gnoise(vec3(cos(ca) * 2.6, sin(ca) * 2.6, rc * 7.0 - d * 0.9)), 5.0)
               * exp(-rc / (0.36 * R));

    // 5. Tendrils: wavy, branching light radiating into the dark (Reference 2).
    float outside = smoothstep(Rb * 0.80, Rb * 1.04, r);
    float tend = 0.0;
    for (int k = 0; k < 3; k++) {
        float fk = float(k);
        if (fk > 1.0 + full) break;
        float wave = 0.55 * gnoise(vec3(dir * 2.2, r * 3.2 + fk * 5.1) + flow * 0.22)
                   + 0.20 * gnoise(vec3(dir * 6.0, r * 7.0 - fk) + flow * 0.4);
        float a2 = ang + wave / max(r, 0.3);
        float spokes = 3.0 + 2.2 * fk;
        vec3 q = vec3(cos(a2) * spokes, sin(a2) * spokes, r * (1.6 + 0.6 * fk) - d * (0.45 + 0.15 * fk)) + fk * 13.7;
        tend += ridge(gnoise(q), 12.0 + 4.0 * fk) * (1.0 - 0.3 * fk);
    }
    tend *= outside * exp(-max(r - Rb, 0.0) * 4.2) * smoothstep(0.99, 0.72, r);

    // 6. Sparks: short bright streaks thrown off the surface (Reference 1).
    float spark = 0.0;
    float a01 = ang / TAU + 0.5;
    for (int k = 0; k < 2; k++) {
        float fk = float(k);
        float cells = 56.0 + 34.0 * fk;
        float ci = floor(a01 * cells);
        vec3 hs = 0.5 + 0.5 * hash33(vec3(ci, fk * 3.1 + 1.0, 7.7));
        float life = fract(hs.x + t * (0.10 + 0.16 * hs.y) + d * 0.05);
        float sr = Rb * (0.97 + 0.85 * life);
        float sa = (ci + 0.5 + (hs.z - 0.5) * 0.7) / cells;
        float da = (a01 - sa) * TAU * r;
        float dr = r - sr;
        float streak = exp(-da * da / 0.000014 - dr * dr / 0.00018);
        spark += streak * (1.0 - life) * step(0.78, hs.y);
    }
    spark *= full;

    // 7. Haze: the translucent body and the deep glow around it.
    float body = inside * (0.06 + 0.20 * smoothstep(Rb, 0.0, r));
    float halo = exp(-max(r - Rb, 0.0) * 7.0) * (1.0 - inside * 0.6);

    // Energy, then colour.
    float standby = 1.0 - 0.55 * u_standby;
    float E = 0.0;
    E += body * 0.50;
    E += fil * (0.20 + 0.95 * u_plasma);
    E += shards * (1.10 + 2.20 * u_plasma);
    E += folds * (0.40 + 1.10 * u_glow);
    E += rim * (1.10 + 2.40 * u_glow);
    E += core * (0.4 + 5.0 * u_heat * u_heat + 1.2 * u_heat);
    E += rays * (0.25 + 1.30 * u_heat);
    E += tend * (0.08 + 0.75 * u_glow * u_glow);
    E += spark * (0.3 + 1.0 * u_glow);
    E += halo * (0.10 + 0.30 * u_glow);
    E *= standby;

    vec3 tint = mix(vec3(0.086, 0.467, 0.824), vec3(0.85, 0.30, 0.20), u_alert * 0.75);  // Electric Blue #1677D2
    vec3 col = 1.0 - exp(-E * tint * 2.1);
    // The hottest light (rim folds, crack crossings, the core) burns toward white.
    col = mix(col, vec3(0.93, 0.98, 1.0), smoothstep(2.2, 6.5, E) * (1.0 - u_alert * 0.6));
    // Midnight Blue #102B4C haze where the light is faint, so the dark around the orb is blue, not grey.
    float haze = halo * (0.55 + 0.45 * u_glow) * standby;
    col += (1.0 - col) * vec3(0.063, 0.169, 0.298) * haze;
    float alpha = clamp(max(col.r, max(col.g, col.b)) * 1.15, 0.0, 1.0);
    col = min(col, vec3(alpha));
    return vec4(col, alpha);
}

// ---- GLSL entry point (Garvis Mobile replaces everything below this line) ----
void main() {
    gl_FragColor = plasma(gl_FragCoord.xy);
}
