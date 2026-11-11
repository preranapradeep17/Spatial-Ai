/**
 * Spatial Quest — Realistic 5-Room Modern Facility & PBR Scene Builder
 *
 * Constructs procedural canvas PBR textures (tile, oak, lab epoxy, drywall, metal),
 * exterior frosted windows, recessed architectural ceiling fixtures, detailed multi-part
 * furniture synchronized with backend Scene Graph coordinates, proximity highlights,
 * and camera-frame capture for live SpatialDetector CNN inference.
 */
(function (global) {
  "use strict";

  function makeProceduralTexture(drawFn, repeatX, repeatY, size) {
    const dim = size || 256;
    const canvas = document.createElement("canvas");
    canvas.width = dim;
    canvas.height = dim;
    const ctx = canvas.getContext("2d");
    drawFn(ctx, dim);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeatX || 1, repeatY || 1);
    tex.anisotropy = 4;
    return tex;
  }

  function createFacilityTextures() {
    // 1. Slate architectural floor tiles (Reception & Lab)
    const slateTileTex = makeProceduralTexture((ctx, s) => {
      ctx.fillStyle = "#1c2430";
      ctx.fillRect(0, 0, s, s);
      ctx.strokeStyle = "#11161f";
      ctx.lineWidth = 3;
      const step = s / 4;
      for (let i = 0; i <= s; i += step) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, s); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(s, i); ctx.stroke();
      }
      for (let k = 0; k < 280; k++) {
        ctx.fillStyle = k % 2 ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.04)";
        ctx.fillRect(Math.random() * s, Math.random() * s, 3, 3);
      }
    }, 6, 4);

    // 2. Hardwood oak planks (Office & Library)
    const woodFloorTex = makeProceduralTexture((ctx, s) => {
      ctx.fillStyle = "#3a2a1d";
      ctx.fillRect(0, 0, s, s);
      const planks = 6;
      const ph = s / planks;
      for (let r = 0; r < planks; r++) {
        const shade = (r % 3) * 6;
        ctx.fillStyle = `rgb(${58 + shade}, ${40 + shade}, ${28 + shade})`;
        ctx.fillRect(0, r * ph, s, ph - 2);
        ctx.strokeStyle = "rgba(18,12,8,0.65)";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, r * ph); ctx.lineTo(s, r * ph); ctx.stroke();
      }
      ctx.fillStyle = "rgba(255,220,180,0.03)";
      for (let k = 0; k < 90; k++) {
        ctx.fillRect(Math.random() * s, Math.random() * s, 28, 1);
      }
    }, 4, 4);

    // 3. Industrial concrete vault floor (Storage Room)
    const concreteTex = makeProceduralTexture((ctx, s) => {
      ctx.fillStyle = "#22272e";
      ctx.fillRect(0, 0, s, s);
      ctx.strokeStyle = "#171b20";
      ctx.lineWidth = 2;
      ctx.strokeRect(1, 1, s - 2, s - 2);
      for (let k = 0; k < 400; k++) {
        ctx.fillStyle = k % 2 ? "rgba(255,255,255,0.025)" : "rgba(0,0,0,0.06)";
        ctx.fillRect(Math.random() * s, Math.random() * s, 2, 2);
      }
    }, 5, 3);

    // 4. Architectural wall panel finish
    const wallPanelTex = makeProceduralTexture((ctx, s) => {
      ctx.fillStyle = "#252f3d";
      ctx.fillRect(0, 0, s, s);
      ctx.strokeStyle = "rgba(15, 23, 42, 0.35)";
      ctx.lineWidth = 2;
      ctx.strokeRect(2, 2, s - 4, s - 4);
      ctx.fillStyle = "rgba(255,255,255,0.015)";
      for (let k = 0; k < 150; k++) {
        ctx.fillRect(Math.random() * s, Math.random() * s, 2, 2);
      }
    }, 6, 1);

    // 5. Oak desk/bookshelf wood surface
    const furnitureWoodTex = makeProceduralTexture((ctx, s) => {
      ctx.fillStyle = "#5c4028";
      ctx.fillRect(0, 0, s, s);
      ctx.fillStyle = "rgba(30,15,5,0.12)";
      for (let k = 0; k < 45; k++) {
        ctx.fillRect(0, Math.random() * s, s, 2);
      }
    }, 1, 1);

    return { slateTileTex, woodFloorTex, concreteTex, wallPanelTex, furnitureWoodTex };
  }

  class FacilityEnvironment {
    constructor(scene) {
      this.scene = scene;
      this.group = new THREE.Group();
      this.group.name = "FacilityRoot";
      this.scene.add(this.group);

      this.textures = createFacilityTextures();
      this.colliders = [];
      this.interactables = []; // { id, title, className, room, position, description, meshGroup, highlightBox }
      this.door3Group = null;
      this.door3Laser = null;
      this.door3Unlocked = false;
      this.hoveredId = null;
      this.selectedId = null;

      this._buildLighting();
      this._buildArchitecture();
      this._buildFurnitureAndClues();
    }

    _buildLighting() {
      const hemi = new THREE.HemisphereLight(0xcbd5e1, 0x1e293b, 0.72);
      this.group.add(hemi);

      const mainDir = new THREE.DirectionalLight(0xe2e8f0, 0.75);
      mainDir.position.set(8, 16, -6);
      mainDir.castShadow = true;
      mainDir.shadow.mapSize.width = 2048;
      mainDir.shadow.mapSize.height = 2048;
      mainDir.shadow.camera.near = 2;
      mainDir.shadow.camera.far = 45;
      mainDir.shadow.camera.left = -16;
      mainDir.shadow.camera.right = 16;
      mainDir.shadow.camera.top = 24;
      mainDir.shadow.camera.bottom = -24;
      mainDir.shadow.bias = -0.0005;
      this.group.add(mainDir);
      this.mainDirLight = mainDir;

      // Zone point lights for warm/cool atmospheric differentiation
      const zoneLights = [
        { pos: [0, 3.4, -15.0], color: 0xfef3c7, intensity: 0.75, dist: 16 }, // Reception warm lobby
        { pos: [6.0, 3.4, -5.0], color: 0xe0f2fe, intensity: 0.65, dist: 13 }, // Office daylight white
        { pos: [-6.0, 3.4, -5.0], color: 0xfde68a, intensity: 0.65, dist: 13 }, // Library study amber
        { pos: [0.0, 3.5, 3.0], color: 0xbae6fd, intensity: 0.85, dist: 16 },  // Lab crisp inspection
        { pos: [0.0, 3.4, 15.5], color: 0xcbd5e1, intensity: 0.65, dist: 16 }, // Storage vault slate
      ];

      zoneLights.forEach((z) => {
        const pl = new THREE.PointLight(z.color, z.intensity, z.dist, 1.5);
        pl.position.set(z[0] || z.pos[0], z.pos[1], z.pos[2]);
        this.group.add(pl);

        // Recessed ceiling fixture panel
        const panel = new THREE.Mesh(
          new THREE.BoxGeometry(1.6, 0.06, 0.7),
          new THREE.MeshBasicMaterial({ color: z.color })
        );
        panel.position.set(z.pos[0], 3.86, z.pos[2]);
        this.group.add(panel);
      });
    }

    _addWallBox(x, y, z, w, h, d, mat, addCollider, tag) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      mesh.position.set(x, y, z);
      mesh.receiveShadow = true;
      mesh.castShadow = true;
      this.group.add(mesh);

      // Subtle architectural dark baseboard trim
      if (y - h * 0.5 < 0.2) {
        const trimMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.7 });
        const trim = new THREE.Mesh(new THREE.BoxGeometry(w + 0.04, 0.14, d + 0.04), trimMat);
        trim.position.set(x, 0.07, z);
        this.group.add(trim);
      }

      if (addCollider) {
        this.colliders.push({
          xmin: x - w * 0.5,
          xmax: x + w * 0.5,
          zmin: z - d * 0.5,
          zmax: z + d * 0.5,
          tag: tag || "wall",
          active: true,
        });
      }
      return mesh;
    }

    _buildArchitecture() {
      const H = 3.9;
      const wallMat = new THREE.MeshStandardMaterial({
        map: this.textures.wallPanelTex,
        roughness: 0.78,
        metalness: 0.08,
      });
      const frameMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.4, metalness: 0.55 });
      const glassMat = new THREE.MeshPhysicalMaterial({
        color: 0x93c5fd,
        transparent: true,
        opacity: 0.2,
        roughness: 0.08,
        metalness: 0.1,
      });

      // Distinct room floors
      const addFloor = (cx, cz, w, d, map, roughness) => {
        const mesh = new THREE.Mesh(
          new THREE.PlaneGeometry(w, d),
          new THREE.MeshStandardMaterial({ map, roughness: roughness || 0.65, metalness: 0.08 })
        );
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(cx, 0, cz);
        mesh.receiveShadow = true;
        this.group.add(mesh);
      };

      addFloor(0, -15.5, 24.4, 11.2, this.textures.slateTileTex, 0.45); // Reception (Z -21..-10)
      addFloor(6.0, -5.0, 12.2, 10.2, this.textures.woodFloorTex, 0.62); // Research Office
      addFloor(-6.0, -5.0, 12.2, 10.2, this.textures.woodFloorTex, 0.65); // Library
      addFloor(0, 4.0, 24.4, 8.2, this.textures.slateTileTex, 0.4); // Lab (Z 0..8)
      addFloor(0, 15.0, 24.4, 14.4, this.textures.concreteTex, 0.85); // Storage Vault (Z 8..22.2)

      // Acoustic architectural ceiling panel grid
      const ceilingTex = makeProceduralTexture((ctx, s) => {
        ctx.fillStyle = "#1e2736";
        ctx.fillRect(0, 0, s, s);
        ctx.strokeStyle = "#141b26";
        ctx.lineWidth = 3;
        const step = s / 4;
        for (let i = 0; i <= s; i += step) {
          ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, s); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(s, i); ctx.stroke();
        }
      }, 8, 12);
      const ceiling = new THREE.Mesh(
        new THREE.PlaneGeometry(24.6, 44.0),
        new THREE.MeshStandardMaterial({
          map: ceilingTex,
          roughness: 0.85,
          emissive: 0x0f172a,
          emissiveIntensity: 0.45,
        })
      );
      ceiling.rotation.x = Math.PI / 2;
      ceiling.position.set(0, H, 1.0);
      this.group.add(ceiling);

      // Outer Perimeter Walls (X = +-12.2, Z = -20.5 .. +22.2)
      this._addWallBox(0, H / 2, -20.5, 24.8, H, 0.35, wallMat, true); // South outer reception wall
      this._addWallBox(0, H / 2, 22.3, 24.8, H, 0.35, wallMat, true);  // North outer vault wall
      this._addWallBox(-12.2, H / 2, 0.9, 0.35, H, 43.2, wallMat, true); // West outer wall
      this._addWallBox(12.2, H / 2, 0.9, 0.35, H, 43.2, wallMat, true);  // East outer wall

      // Exterior recessed glass windows along East & West walls
      [-15.0, -5.0, 4.0].forEach((wz) => {
        [-11.98, 11.98].forEach((wx) => {
          const wPane = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.65, 3.4), glassMat);
          wPane.position.set(wx, 1.95, wz);
          this.group.add(wPane);
          const sill = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.08, 3.55), frameMat);
          sill.position.set(wx, 1.1, wz);
          this.group.add(sill);
        });
      });

      // Partition Z = -10 (Reception -> Central Corridor / Office / Library)
      // Central wide doorway gap X in [-2.2, +2.2]
      this._addWallBox(-7.2, H / 2, -10.0, 10.0, H, 0.28, wallMat, true);
      this._addWallBox(7.2, H / 2, -10.0, 10.0, H, 0.28, wallMat, true);
      this._addWallBox(0, 3.45, -10.0, 4.6, 0.9, 0.28, frameMat, false); // Transom header

      // Central North-South corridor flanking walls between Library (X < -2.2) & Office (X > 2.2)
      // with framed doorway openings into Library (X=-2.2, Z=-5) & Office (X=+2.2, Z=-5)
      [-2.2, 2.2].forEach((cx) => {
        this._addWallBox(cx, H / 2, -8.35, 0.22, H, 3.3, wallMat, true);
        this._addWallBox(cx, H / 2, -1.65, 0.22, H, 3.3, wallMat, true);
        // Glass interior window panes looking into Office/Library
        const gp = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.4, 2.2), glassMat);
        gp.position.set(cx, 1.85, -8.35);
        this.group.add(gp);
        this._addWallBox(cx, 3.4, -5.0, 0.24, 1.0, 3.4, frameMat, false);
      });

      // Partition Z = 0 (Corridor -> Investigation Laboratory), central portal X in [-2.4, +2.4]
      this._addWallBox(-7.3, H / 2, 0.0, 9.8, H, 0.28, wallMat, true);
      this._addWallBox(7.3, H / 2, 0.0, 9.8, H, 0.28, wallMat, true);
      this._addWallBox(0, 3.45, 0.0, 5.0, 0.9, 0.28, frameMat, false);

      // Partition Z = 8.0 (Investigation Lab -> Storage Vault, Security Door 3 at X in [-1.8, +1.8])
      this._addWallBox(-7.0, H / 2, 8.0, 10.2, H, 0.3, wallMat, true);
      this._addWallBox(7.0, H / 2, 8.0, 10.2, H, 0.3, wallMat, true);
      this._addWallBox(0, 3.35, 8.0, 3.8, 1.1, 0.32, frameMat, false);
      // Lab observation glass windows along Z=8
      [-5.2, 5.2].forEach((gx) => {
        const labWin = new THREE.Mesh(new THREE.BoxGeometry(3.2, 1.4, 0.08), glassMat);
        labWin.position.set(gx, 1.9, 8.0);
        this.group.add(labWin);
      });

      // Security Door 3 Movable Panel + Keypad
      const door3Grp = new THREE.Group();
      door3Grp.position.set(0, 0, 8.0);
      const doorLeaf = new THREE.Mesh(
        new THREE.BoxGeometry(3.45, 2.75, 0.14),
        new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.35, metalness: 0.72 })
      );
      doorLeaf.position.set(0, 1.38, 0);
      doorLeaf.castShadow = true;
      door3Grp.add(doorLeaf);

      const lockBar = new THREE.Mesh(
        new THREE.BoxGeometry(3.2, 0.08, 0.18),
        new THREE.MeshStandardMaterial({ color: 0xf43f5e, emissive: 0xe11d48, emissiveIntensity: 0.75 })
      );
      lockBar.position.set(0, 1.45, 0);
      door3Grp.add(lockBar);
      this.group.add(door3Grp);
      this.door3Group = door3Grp;
      this.door3Laser = lockBar;

      // Collider blocking Door 3 until unlocked
      this.colliders.push({ xmin: -1.8, xmax: 1.8, zmin: 7.7, zmax: 8.3, tag: "door_3", active: true });

      // Door 3 Keypad Console mounted on wall beside Door 3
      const keypad = new THREE.Mesh(
        new THREE.BoxGeometry(0.28, 0.42, 0.12),
        new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.4, metalness: 0.6 })
      );
      keypad.position.set(2.05, 1.35, 7.82);
      this.group.add(keypad);
      const keypadScreen = new THREE.Mesh(
        new THREE.PlaneGeometry(0.18, 0.14),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
      );
      keypadScreen.position.set(2.05, 1.43, 7.75);
      keypadScreen.rotation.y = Math.PI;
      this.group.add(keypadScreen);
    }

    setDoor3Unlocked(unlocked) {
      this.door3Unlocked = Boolean(unlocked);
      if (this.door3Group) {
        this.door3Group.position.x = this.door3Unlocked ? -3.2 : 0.0;
      }
      if (this.door3Laser) {
        this.door3Laser.material.color.setHex(this.door3Unlocked ? 0x10b981 : 0xf43f5e);
        this.door3Laser.material.emissive.setHex(this.door3Unlocked ? 0x059669 : 0xe11d48);
      }
      this.colliders.forEach((c) => {
        if (c.tag === "door_3") c.active = !this.door3Unlocked;
      });
    }

    _registerInteractable(meta, meshGroup, colliderHalfExtents) {
      meshGroup.userData.interactId = meta.id;
      meshGroup.traverse((c) => { c.userData.interactId = meta.id; });
      this.group.add(meshGroup);

      // Architectural subtle highlight ring on floor
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.65, 0.72, 32),
        new THREE.MeshBasicMaterial({
          color: 0x38bdf8,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.0,
        })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(meta.position[0], 0.03, meta.position[2]);
      this.group.add(ring);

      if (colliderHalfExtents) {
        const [hx, hz] = colliderHalfExtents;
        this.colliders.push({
          xmin: meta.position[0] - hx,
          xmax: meta.position[0] + hx,
          zmin: meta.position[2] - hz,
          zmax: meta.position[2] + hz,
          tag: meta.id,
          active: true,
        });
      }

      this.interactables.push({
        id: meta.id,
        title: meta.title,
        className: meta.className,
        room: meta.room,
        position: meta.position,
        confidence: meta.confidence || 0.91,
        depth: meta.depth || 0.0,
        source: "facility_layout",
        description: meta.description,
        meshGroup,
        ring,
      });
    }

    _makeTable(w, h, d, topColor, legColor) {
      const g = new THREE.Group();
      const topMat = new THREE.MeshStandardMaterial({
        map: this.textures.furnitureWoodTex,
        color: topColor || 0x8b6b4a,
        roughness: 0.55,
      });
      const legMat = new THREE.MeshStandardMaterial({ color: legColor || 0x1e293b, roughness: 0.4, metalness: 0.65 });
      const top = new THREE.Mesh(new THREE.BoxGeometry(w, 0.07, d), topMat);
      top.position.y = h;
      top.castShadow = true; top.receiveShadow = true;
      g.add(top);

      const lx = w * 0.5 - 0.1;
      const lz = d * 0.5 - 0.1;
      [[-lx, -lz], [lx, -lz], [-lx, lz], [lx, lz]].forEach(([px, pz]) => {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, h, 8), legMat);
        leg.position.set(px, h * 0.5, pz);
        leg.castShadow = true;
        g.add(leg);
      });
      return g;
    }

    _makeChair(seatColor) {
      const g = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color: seatColor || 0x263244, roughness: 0.7 });
      const frame = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.4, metalness: 0.7 });
      const seat = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.07, 0.52), mat);
      seat.position.y = 0.46;
      seat.castShadow = true;
      g.add(seat);
      const back = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.52, 0.07), mat);
      back.position.set(0, 0.76, -0.23);
      back.castShadow = true;
      g.add(back);
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.04, 0.43, 8), frame);
      stem.position.y = 0.22;
      g.add(stem);
      const base = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.04, 12), frame);
      base.position.y = 0.02;
      g.add(base);
      return g;
    }

    _makeBookshelf() {
      const g = new THREE.Group();
      const woodMat = new THREE.MeshStandardMaterial({ map: this.textures.furnitureWoodTex, color: 0x5d4037, roughness: 0.65 });
      const back = new THREE.Mesh(new THREE.BoxGeometry(0.12, 2.2, 2.0), woodMat);
      back.position.set(-0.18, 1.1, 0);
      back.castShadow = true;
      g.add(back);
      [0.08, 0.6, 1.12, 1.64, 2.15].forEach((sy) => {
        const shelf = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.06, 2.0), woodMat);
        shelf.position.set(0, sy, 0);
        shelf.castShadow = true;
        g.add(shelf);
      });
      const palette = [0x3b82f6, 0x94a3b8, 0xd97706, 0x475569, 0x0f766e, 0x7f1d1d];
      [0.24, 0.76, 1.28, 1.8].forEach((by, rIdx) => {
        for (let i = 0; i < 6; i++) {
          const bk = new THREE.Mesh(
            new THREE.BoxGeometry(0.28, 0.34, 0.24),
            new THREE.MeshStandardMaterial({ color: palette[(i + rIdx) % palette.length], roughness: 0.8 })
          );
          bk.position.set(0.02, by, -0.78 + i * 0.31);
          g.add(bk);
        }
      });
      return g;
    }

    _makeCabinet() {
      const g = new THREE.Group();
      const steel = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.38, metalness: 0.65 });
      const dark = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.4, metalness: 0.7 });
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.85, 1.6, 1.1), steel);
      body.position.y = 0.8;
      body.castShadow = true; body.receiveShadow = true;
      g.add(body);
      [-0.25, 0.25].forEach((dz) => {
        const handle = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.22, 0.04), dark);
        handle.position.set(-0.44, 0.9, dz);
        g.add(handle);
      });
      return g;
    }

    _makeWorkstationRig(parent) {
      const darkMetal = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.3, metalness: 0.7 });
      const screenMat = new THREE.MeshBasicMaterial({ color: 0x0ea5e9 });
      const mon = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.42, 0.04), darkMetal);
      mon.position.set(0, 1.06, -0.2);
      parent.add(mon);
      const scr = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.36), screenMat);
      scr.position.set(0, 1.06, -0.175);
      parent.add(scr);
      const stand = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.22, 0.1), darkMetal);
      stand.position.set(0, 0.86, -0.22);
      parent.add(stand);
    }

    _buildFurnitureAndClues() {
      // ── 1. RECEPTION / CENTRAL HUB ─────────────────────────────────────
      const recTable = this._makeTable(2.2, 0.76, 1.1, 0x6b4c35, 0x1e293b);
      this._makeWorkstationRig(recTable);
      recTable.position.set(0.0, 0.0, -15.0);
      this._registerInteractable({
        id: "table_reception",
        title: "Reception Security Table",
        className: "table",
        room: "Reception / Central Hub",
        position: [0.0, 0.75, -15.0],
        description: "Primary reception security desk with logbook workstation. Staff logged missing key incident here.",
      }, recTable, [1.15, 0.65]);

      const recChair = this._makeChair(0x1f2937);
      recChair.position.set(0.0, 0.0, -13.8);
      recChair.rotation.y = Math.PI;
      this._registerInteractable({
        id: "chair_reception",
        title: "Reception Swivel Chair",
        className: "chair",
        room: "Reception / Central Hub",
        position: [0.0, 0.45, -13.8],
        description: "Ergonomic security attendant chair positioned behind Reception Security Table.",
      }, recChair, [0.35, 0.35]);

      const sofa = new THREE.Group();
      const leather = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.75 });
      const sofaBase = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.42, 0.92), leather);
      sofaBase.position.y = 0.22; sofaBase.castShadow = true;
      const sofaBack = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.55, 0.22), leather);
      sofaBack.position.set(0, 0.62, -0.35); sofaBack.castShadow = true;
      sofa.add(sofaBase, sofaBack);
      sofa.position.set(-6.5, 0.0, -16.0);
      this._registerInteractable({
        id: "sofa_visitor",
        title: "Visitor Lounge Sofa",
        className: "sofa",
        room: "Reception / Central Hub",
        position: [-6.5, 0.3, -16.0],
        description: "Modern upholstered lobby seating in the west corner of Reception.",
      }, sofa, [1.25, 0.55]);

      const termTable = this._makeTable(1.6, 0.76, 0.9, 0x475569, 0x0f172a);
      this._makeWorkstationRig(termTable);
      termTable.position.set(6.5, 0.0, -14.0);
      this._registerInteractable({
        id: "table_terminal",
        title: "Facility Central Terminal",
        className: "table",
        room: "Reception / Central Hub",
        position: [6.5, 0.75, -14.0],
        description: "Central diagnostics workstation controlling building ventilation and emergency status.",
      }, termTable, [0.85, 0.5]);

      // ── 2. RESEARCH OFFICE (X >= 2.2, Z in [-10, 0]) ───────────────────
      const offTable = this._makeTable(1.8, 0.76, 1.0, 0x7c5836, 0x1e293b);
      this._makeWorkstationRig(offTable);
      offTable.position.set(6.0, 0.0, -5.0);
      this._registerInteractable({
        id: "table_office_desk",
        title: "Research Office Desk",
        className: "table",
        room: "Research Office",
        position: [6.0, 0.75, -5.0],
        description: "Senior researcher workstation equipped with incident dossier display.",
      }, offTable, [0.95, 0.55]);

      const offChair = this._makeChair(0x273549);
      offChair.position.set(6.0, 0.0, -4.0);
      offChair.rotation.y = Math.PI;
      this._registerInteractable({
        id: "chair_office",
        title: "Research Office Chair",
        className: "chair",
        room: "Research Office",
        position: [6.0, 0.45, -4.0],
        description: "Workstation task chair in the Research Office.",
      }, offChair, [0.35, 0.35]);

      const offCab = this._makeCabinet();
      offCab.position.set(9.0, 0.0, -5.0);
      this._registerInteractable({
        id: "cabinet_office_files",
        title: "Office Filing Cabinet",
        className: "cabinet",
        room: "Research Office",
        position: [9.0, 1.2, -5.0],
        description: "Locked steel filing cabinet storing personnel records and access forms.",
      }, offCab, [0.5, 0.6]);

      // ── 3. LIBRARY / ARCHIVES (X <= -2.2, Z in [-10, 0]) ───────────────
      const libShelf = this._makeBookshelf();
      libShelf.position.set(-8.5, 0.0, -5.0);
      this._registerInteractable({
        id: "bookshelf_library_reference",
        title: "Library Reference Bookshelf",
        className: "bookshelf",
        room: "Library / Archives",
        position: [-8.5, 1.6, -5.0],
        description: "Tall archive shelving containing maintenance blueprints and incident manuals.",
      }, libShelf, [0.4, 1.05]);

      const libTable = this._makeTable(1.8, 0.76, 1.0, 0x6d4c33, 0x1e293b);
      libTable.position.set(-4.5, 0.0, -5.0);
      this._registerInteractable({
        id: "table_library_reading",
        title: "Library Reading Table",
        className: "table",
        room: "Library / Archives",
        position: [-4.5, 0.75, -5.0],
        description: "Quiet reading table in the Library & Archives wing.",
      }, libTable, [0.95, 0.55]);

      const libChair = this._makeChair(0x3f2e21);
      libChair.position.set(-4.5, 0.0, -4.0);
      libChair.rotation.y = Math.PI;
      this._registerInteractable({
        id: "chair_library",
        title: "Library Reading Chair",
        className: "chair",
        room: "Library / Archives",
        position: [-4.5, 0.45, -4.0],
        description: "Study chair next to the Library reading table.",
      }, libChair, [0.35, 0.35]);

      // ── 4. INVESTIGATION LABORATORY (Z in [0, 8]) ───────────────────────
      const labTable = this._makeTable(2.3, 0.76, 1.2, 0x334155, 0x0f172a);
      // Lab glassware & Microscope details on lab table
      const flask = new THREE.Mesh(
        new THREE.ConeGeometry(0.08, 0.22, 12),
        new THREE.MeshPhysicalMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.65, roughness: 0.1 })
      );
      flask.position.set(-0.55, 0.89, 0.1);
      labTable.add(flask);

      // Researcher's Forgotten Datapad (clue_pda_02) placed right on top of table_lab
      const pdaMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.24, 0.025, 0.32),
        new THREE.MeshStandardMaterial({ color: 0x0f172a, emissive: 0xf59e0b, emissiveIntensity: 0.35 })
      );
      pdaMesh.position.set(0.55, 0.81, 0.2);
      labTable.add(pdaMesh);

      labTable.position.set(0.0, 0.0, 0.0);
      this._registerInteractable({
        id: "table_lab",
        title: "Central Lab Workbench (+ Datapad)",
        className: "table",
        room: "Investigation Laboratory",
        position: [0.0, 0.75, 0.0],
        description: "Main chemical and sensor workbench. Holds containment glassware and Researcher's Forgotten Datapad.",
      }, labTable, [1.2, 0.65]);

      const labChair = this._makeChair(0x1e293b);
      labChair.position.set(-1.4, 0.0, 0.6);
      labChair.rotation.y = Math.PI * 0.4;
      this._registerInteractable({
        id: "chair_lab",
        title: "Lab Workbench Chair",
        className: "chair",
        room: "Investigation Laboratory",
        position: [-1.4, 0.45, 0.6],
        description: "Adjustable lab chair located left/north-west of the Central Lab Workbench.",
      }, labChair, [0.35, 0.35]);

      const labCab = this._makeCabinet();
      labCab.position.set(8.5, 0.0, 2.0);
      this._registerInteractable({
        id: "cabinet_lab",
        title: "Laboratory Steel Cabinet",
        className: "cabinet",
        room: "Investigation Laboratory",
        position: [8.5, 1.2, 2.0],
        description: "Heavy laboratory steel cabinet. Maintenance logs mention a physical backup Research Key stored here.",
      }, labCab, [0.5, 0.6]);

      const labShelf = this._makeBookshelf();
      labShelf.position.set(-8.5, 0.0, 2.0);
      labShelf.rotation.y = Math.PI;
      this._registerInteractable({
        id: "bookshelf_lab",
        title: "Laboratory Reference Shelf",
        className: "bookshelf",
        room: "Investigation Laboratory",
        position: [-8.5, 1.6, 2.0],
        description: "Reference binders and optical calibration manuals on the west wall of the lab.",
      }, labShelf, [0.4, 1.05]);

      // Door 3 interactable marker right at keypad [0, 2.0, 8.0]
      const doorMarker = new THREE.Group();
      doorMarker.position.set(0.0, 0.0, 8.0);
      const badge = new THREE.Mesh(
        new THREE.BoxGeometry(0.42, 0.14, 0.06),
        new THREE.MeshBasicMaterial({ color: 0xf43f5e })
      );
      badge.position.set(2.05, 1.68, -0.2);
      doorMarker.add(badge);
      this._registerInteractable({
        id: "door_3",
        title: "Security Door 3 (Vault Gate)",
        className: "door",
        room: "Investigation Laboratory",
        position: [0.0, 2.0, 8.0],
        description: "Electronic security gate separating Lab and Storage Vault. Requires 4-digit maintenance override passcode.",
      }, doorMarker, null);

      // ── 5. STORAGE ROOM / VAULT (Z > 8) ────────────────────────────────
      const archA = this._makeBookshelf();
      archA.position.set(-8.5, 0.0, 14.0);
      archA.rotation.y = Math.PI;
      this._registerInteractable({
        id: "bookshelf_archive_a",
        title: "Vault Archive Rack A",
        className: "bookshelf",
        room: "Storage Room",
        position: [-8.5, 1.6, 14.0],
        description: "High-security archival shelving Unit A inside the Storage Vault.",
      }, archA, [0.4, 1.05]);

      const archB = this._makeBookshelf();
      archB.position.set(-8.5, 0.0, 18.0);
      archB.rotation.y = Math.PI;
      this._registerInteractable({
        id: "bookshelf_archive_b",
        title: "Vault Archive Rack B",
        className: "bookshelf",
        room: "Storage Room",
        position: [-8.5, 1.6, 18.0],
        description: "High-security archival shelving Unit B inside the Storage Vault.",
      }, archB, [0.4, 1.05]);

      const vaultCab = this._makeCabinet();
      vaultCab.position.set(8.5, 0.0, 14.0);
      this._registerInteractable({
        id: "cabinet_vault",
        title: "Armored Vault Locker",
        className: "cabinet",
        room: "Storage Room",
        position: [8.5, 1.2, 14.0],
        description: "Reinforced storage locker inside Sector 5 Vault.",
      }, vaultCab, [0.5, 0.6]);

      const exitDoor = new THREE.Group();
      exitDoor.position.set(0.0, 0.0, 22.0);
      const exitFrame = new THREE.Mesh(
        new THREE.BoxGeometry(2.6, 2.6, 0.25),
        new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.4, metalness: 0.7 })
      );
      exitFrame.position.y = 1.3;
      const exitSign = new THREE.Mesh(
        new THREE.BoxGeometry(0.9, 0.26, 0.08),
        new THREE.MeshBasicMaterial({ color: 0x10b981 })
      );
      exitSign.position.set(0, 2.75, -0.15);
      exitDoor.add(exitFrame, exitSign);
      this._registerInteractable({
        id: "door_emergency_exit",
        title: "Emergency Exit Door",
        className: "door",
        room: "Storage Room",
        position: [0.0, 2.0, 22.0],
        description: "Reinforced north emergency egress door with illuminated status indicator.",
      }, exitDoor, [1.3, 0.2]);
    }

    getNearestInteractable(playerPos, maxDist) {
      const limit = maxDist || 3.0;
      let best = null;
      let bestD = Infinity;
      this.interactables.forEach((item) => {
        const d = Math.hypot(item.position[0] - playerPos.x, item.position[2] - playerPos.z);
        if (d <= limit && d < bestD) {
          bestD = d;
          best = { item, distance: d };
        }
      });
      return best;
    }

    updateHighlights(hoveredId, selectedId, nearId) {
      this.hoveredId = hoveredId;
      this.selectedId = selectedId;
      this.interactables.forEach((item) => {
        if (!item.ring) return;
        const isSel = item.id === selectedId;
        const isHov = item.id === hoveredId;
        const isNear = item.id === nearId;
        if (isSel) {
          item.ring.material.color.setHex(0xf59e0b);
          item.ring.material.opacity = 0.85;
        } else if (isHov || isNear) {
          item.ring.material.color.setHex(0x38bdf8);
          item.ring.material.opacity = 0.55;
        } else {
          item.ring.material.opacity = 0.0;
        }
      });
    }

    getInteractableById(id) {
      return this.interactables.find((i) => i.id === id) || null;
    }

    getRoomAt(x, z) {
      if (z < -10.0) return "Reception / Central Hub";
      if (z < 0.0) return x >= 0.0 ? "Research Office" : "Library / Archives";
      if (z <= 8.5) return "Investigation Laboratory";
      return "Storage Room";
    }
  }

  global.SpatialFacility3D = { FacilityEnvironment };
})(window);
