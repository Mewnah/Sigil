//! Overlay transforms (OpenVR `HmdMatrix34_t`: three rows of `[rotation | translation]`, metres,
//! right-handed, +Y up, -Z forward). The overlay quad lies in its local XY plane and is seen from
//! its +Z side, so every transform turns +Z toward the user's eyes.

/// Row-major 3x4 matrix (`HmdMatrix34_t.m`).
pub(crate) type Mat34 = [[f32; 4]; 3];

#[cfg(test)]
pub(crate) const IDENTITY: Mat34 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];

/// Pitch (radians, around +X) that turns the quad's +Z toward an eye `offset_y` above/below and
/// `distance` in front of it.
fn pitch_toward_eye(distance: f32, offset_y: f32) -> f32 {
    offset_y.atan2(distance.max(1e-3))
}

/// `Ry(yaw) * Rx(pitch)` with a translation.
fn yaw_pitch(yaw: f32, pitch: f32, t: [f32; 3]) -> Mat34 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    [[cy, sy * sp, sy * cp, t[0]], [0.0, cp, -sp, t[1]], [-sy, cy * sp, cy * cp, t[2]]]
}

/// Head attach: `distance` metres in front of the eyes and `offset_y` above (negative: below the line
/// of sight), tilted to face the eyes. Relative to the HMD (`k_unTrackedDeviceIndex_Hmd`).
pub(crate) fn head(distance: f32, offset_y: f32) -> Mat34 {
    yaw_pitch(0.0, pitch_toward_eye(distance, offset_y), [0.0, offset_y, -distance])
}

/// Hand attach: just above the controller, tilted up and back toward the user's face (relative to
/// the controller's tracked device).
pub(crate) fn hand() -> Mat34 {
    yaw_pitch(0.0, (-50.0f32).to_radians(), [0.0, 0.07, 0.02])
}

/// World attach: placed in the play space (standing universe) `distance` in front of where the
/// headset looks now and `offset_y` above eye height, upright and facing the user. `hmd` is the
/// headset's device-to-absolute pose.
pub(crate) fn world(hmd: &Mat34, distance: f32, offset_y: f32) -> Mat34 {
    let pos = [hmd[0][3], hmd[1][3], hmd[2][3]];
    // Forward is the headset's -Z column, flattened onto the floor plane. Looking straight up or
    // down leaves no horizontal forward: the headset's up (Y) column then points the way the face
    // is turned (sign by where the face points).
    let fwd = [-hmd[0][2], -hmd[2][2]];
    let up = [hmd[0][1], hmd[2][1]];
    let looking_down = -hmd[1][2] < 0.0;
    let mut flat = if (fwd[0] * fwd[0] + fwd[1] * fwd[1]).sqrt() > 1e-3 {
        fwd
    } else if looking_down {
        up
    } else {
        [-up[0], -up[1]]
    };
    let len = (flat[0] * flat[0] + flat[1] * flat[1]).sqrt();
    if len > 1e-6 && len.is_finite() {
        flat = [flat[0] / len, flat[1] / len];
    } else {
        flat = [0.0, -1.0];
    }
    let center = [pos[0] + flat[0] * distance, pos[1] + offset_y, pos[2] + flat[1] * distance];
    // The quad's +Z must point back at the user: (sin yaw, cos yaw) = -flat.
    let yaw = (-flat[0]).atan2(-flat[1]);
    yaw_pitch(yaw, pitch_toward_eye(distance, offset_y), center)
}

/// `m * (x, y, z)` as a direction (no translation).
#[cfg(test)]
pub(crate) fn rotate(m: &Mat34, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    }

    #[test]
    fn head_faces_the_eyes() {
        let m = head(1.2, -0.25);
        assert_eq!([m[0][3], m[1][3], m[2][3]], [0.0, -0.25, -1.2]);
        // The quad's normal points from its centre to the eyes at the origin.
        let to_eye = normalize([0.0, 0.25, 1.2]);
        assert!(close(rotate(&m, [0.0, 0.0, 1.0]), to_eye));
        // Upright: local X stays horizontal.
        assert!(close(rotate(&m, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]));
        // Straight ahead without an offset: no tilt.
        assert!(close(rotate(&head(1.0, 0.0), [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn world_is_in_front_and_upright() {
        // Headset at (1, 1.6, 2) turned 90 degrees left (looking along -X).
        let yaw = std::f32::consts::FRAC_PI_2;
        let (s, c) = yaw.sin_cos();
        let hmd: Mat34 = [[c, 0.0, s, 1.0], [0.0, 1.0, 0.0, 1.6], [-s, 0.0, c, 2.0]];
        let m = world(&hmd, 1.5, -0.3);
        let center = [m[0][3], m[1][3], m[2][3]];
        assert!(close(center, [1.0 - 1.5, 1.3, 2.0]), "{center:?}");
        // Faces back toward the headset (+X), tilted up toward the eyes.
        let n = rotate(&m, [0.0, 0.0, 1.0]);
        assert!(close(n, normalize([1.5, 0.3, 0.0])), "{n:?}");
        // Upright: local X stays horizontal.
        assert!(rotate(&m, [1.0, 0.0, 0.0])[1].abs() < 1e-5);
    }

    #[test]
    fn world_handles_looking_straight_down() {
        // Pitched 90 degrees down: -Z points at the floor, +Y points forward (-Z of the room).
        let hmd: Mat34 = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.7], [0.0, -1.0, 0.0, 0.0]];
        let m = world(&hmd, 1.0, 0.0);
        assert!(close([m[0][3], m[1][3], m[2][3]], [0.0, 1.7, -1.0]));
        assert!(close(rotate(&m, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]));
        assert!(m.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn hand_faces_up_and_back() {
        let n = rotate(&hand(), [0.0, 0.0, 1.0]);
        assert!(n[1] > 0.5 && n[2] > 0.5, "{n:?}");
    }
}
