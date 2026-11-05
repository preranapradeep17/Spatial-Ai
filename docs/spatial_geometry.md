# Spatial Geometry & 3D Localization Reference

## 1. Pinhole Camera Coordinate Frames & Projections

### Optical Camera Frame (Standard CV Convention)
- Origin $(0, 0, 0)$ is at the camera optical center.
- $+X$ points **Right** across the image.
- $+Y$ points **Down** along the image column.
- $+Z$ points **Forward** along the optical axis into the scene.

### Unity Coordinate Frame
- Left-handed Cartesian coordinate system:
- $+X$ points **Right**
- $+Y$ points **Up**
- $+Z$ points **Forward**

### Camera Optical to Unity Local Frame
To bridge optical coordinates $(X_c, Y_c, Z_c)$ into Unity camera local coordinates $(X_u, Y_u, Z_u)$:
$$X_u = X_c$$
$$Y_u = -Y_c \quad (\text{since image } Y \text{ points down, while Unity } Y \text{ points up})$$
$$Z_u = Z_c$$

---

## 2. 2D-to-3D Back-Projection Mathematics
Given a pixel coordinate $(u, v)$ on the image plane and its corresponding linear depth $Z = \text{depth}(u, v)$ in meters:

$$X_{cam} = \frac{(u - c_x) \cdot Z}{f_x}$$
$$Y_{cam} = \frac{(v - c_y) \cdot Z}{f_y}$$
$$Z_{cam} = Z$$

where:
- $f_x, f_y$ are horizontal and vertical focal lengths in pixel units.
- $c_x, c_y$ are principal point pixel coordinates (typically image center $\approx \frac{W}{2}, \frac{H}{2}$).

---

## 3. Robust Depth Sampling
Rather than sampling depth from an isolated single pixel $(u_{center}, v_{center})$ which can be distorted by occlusions, edge bleed, or empty pixels:
1. Define the inner region of interest (inner $40\%$ of bounding box):
   $$\Delta x = 0.4 \cdot (x_{max} - x_{min}), \quad \Delta y = 0.4 \cdot (y_{max} - y_{min})$$
2. Extract depth patch $P \subset \text{DepthMap}$.
3. Filter out invalid depths ($Z \le 0.1\text{m}$, $\text{NaN}$, $\infty$).
4. Compute $Z_{robust} = \text{median}(P_{valid})$.

---

## 4. Camera-to-World Coordinate Transform
Using the camera's world position $T_{cam} \in \mathbb{R}^3$ and rotation matrix $R_{cam} \in \mathbb{R}^{3 \times 3}$ derived from its orientation quaternion $q = [x, y, z, w]$:

$$P_{world} = R_{cam} \cdot \begin{bmatrix} X_c \\ -Y_c \\ Z_c \end{bmatrix} + T_{cam}$$

---

## 5. Spatial Relations Formulation
- **3D Euclidean Distance**:
  $$d(A, B) = \sqrt{(X_A - X_B)^2 + (Y_A - Y_B)^2 + (Z_A - Z_B)^2}$$
- **Proximity ("near")**:
  $$\text{near}(A, B) \iff d(A, B) \le 2.0\text{ meters}$$
- **Support ("on")**:
  $$\text{on}(A, B) \iff 0.10 \le (Y_A - Y_B) \le 0.80 \quad \text{and} \quad \sqrt{(X_A - X_B)^2 + (Z_A - Z_B)^2} \le 0.8\text{m}$$
- **Directional ("left_of" / "right_of")**:
  $$\text{left\_of}(A, B) \iff (X_A - X_B) \le -0.30\text{m} \quad \text{and} \quad |Z_A - Z_B| \le 1.5\text{m}$$
  $$\text{right\_of}(A, B) \iff (X_A - X_B) \ge +0.30\text{m} \quad \text{and} \quad |Z_A - Z_B| \le 1.5\text{m}$$
