/**
 * Spatial Quest — Human Investigator 3D Character Controller
 *
 * Integrates the rigged humanoid GLB model (/assets/investigator.glb, Mixamo humanoid
 * skeleton with Idle, Walk, Run animations), smooth skeletal crossfading, investigative
 * outfit shading, 1st/3rd/top-down camera following, and AABB sliding collisions.
 * Provides an articulated humanoid fallback when loaded offline over file:// CORS.
 */
(function (global) {
  "use strict";

  function wrapAngle(a) {
    while (a > Math.PI) a -= Math.PI * 2;
    while (a < -Math.PI) a += Math.PI * 2;
    return a;
  }

  class InvestigatorController {
    constructor(scene, initialPos) {
      this.scene = scene;
      this.root = new THREE.Group();
      this.root.name = "InvestigatorRoot";
      this.position = new THREE.Vector3(
        initialPos ? initialPos.x : 0.0,
        0.0,
        initialPos ? initialPos.z : -15.0
      );
      this.root.position.copy(this.position);
      this.scene.add(this.root);

      this.facingYaw = Math.PI; // Start facing North (+Z into facility)
      this.cameraYaw = Math.PI;
      this.cameraPitch = -0.18;
      this.cameraMode = "3rd"; // "3rd" | "1st" | "top"
      this.cameraPosSmooth = new THREE.Vector3(0, 2.5, -20.1);
      this.cameraLookSmooth = new THREE.Vector3(0, 1.4, -15.0);

      this.mixer = null;
      this.actions = {};
      this.activeActionName = "Idle";
      this.modelGroup = null;
      this.isGLBLoaded = false;
      this.fallbackRig = null;

      this.walkSpeed = 4.6;
      this.runSpeed = 7.8;
      this.radius = 0.42;
      this.colliders = []; // [{xmin, xmax, zmin, zmax, tag, active}]
      this.interactTimer = 0.0;
      this.isMoving = false;
      this.isRunning = false;
      this.scannerLight = null;
      this.rightHandBone = null;

      this._createFallbackHumanoid();
    }

    /**
     * Builds an articulated detective humanoid fallback used while GLB streams
     * or when restricted by local file:// browser CORS.
     */
    _createFallbackHumanoid() {
      const rig = new THREE.Group();
      rig.name = "FallbackHumanoidRig";

      const jacketMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        roughness: 0.72,
        metalness: 0.12,
      });
      const trousersMat = new THREE.MeshStandardMaterial({
        color: 0x0f172a,
        roughness: 0.82,
        metalness: 0.05,
      });
      const skinMat = new THREE.MeshStandardMaterial({
        color: 0xd8a888,
        roughness: 0.65,
        metalness: 0.02,
      });
      const hairMat = new THREE.MeshStandardMaterial({
        color: 0x1b1918,
        roughness: 0.85,
      });
      const shoeMat = new THREE.MeshStandardMaterial({
        color: 0x141414,
        roughness: 0.55,
        metalness: 0.15,
      });
      const shirtMat = new THREE.MeshStandardMaterial({
        color: 0xcbd5e1,
        roughness: 0.7,
      });

      const torso = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.56, 0.24), jacketMat);
      torso.position.y = 1.18;
      torso.castShadow = true;
      rig.add(torso);

      const collar = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.08, 0.18), shirtMat);
      collar.position.set(0, 1.46, -0.02);
      rig.add(collar);

      const belt = new THREE.Mesh(new THREE.BoxGeometry(0.43, 0.06, 0.25), shoeMat);
      belt.position.y = 0.92;
      rig.add(belt);

      const head = new THREE.Mesh(new THREE.SphereGeometry(0.135, 16, 16), skinMat);
      head.position.y = 1.63;
      head.castShadow = true;
      rig.add(head);

      const hair = new THREE.Mesh(new THREE.SphereGeometry(0.142, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.56), hairMat);
      hair.position.set(0, 1.64, 0.01);
      rig.add(hair);

      const makeLimb = (w, h, d, mat, x, y) => {
        const pivot = new THREE.Group();
        pivot.position.set(x, y, 0);
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
        mesh.position.y = -h * 0.5;
        mesh.castShadow = true;
        pivot.add(mesh);
        rig.add(pivot);
        return pivot;
      };

      const leftArm = makeLimb(0.11, 0.52, 0.11, jacketMat, -0.28, 1.42);
      const rightArm = makeLimb(0.11, 0.52, 0.11, jacketMat, 0.28, 1.42);
      const leftLeg = makeLimb(0.15, 0.78, 0.16, trousersMat, -0.11, 0.88);
      const rightLeg = makeLimb(0.15, 0.78, 0.16, trousersMat, 0.11, 0.88);

      const scanner = new THREE.Mesh(
        new THREE.BoxGeometry(0.06, 0.04, 0.14),
        new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0ea5e9, emissiveIntensity: 0.4 })
      );
      scanner.position.set(0, -0.52, -0.05);
      rightArm.add(scanner);

      this.fallbackRig = { rig, leftArm, rightArm, leftLeg, rightLeg, phase: 0 };
      this.root.add(rig);
    }

    /**
     * Asynchronously loads the rigged Mixamo GLB investigator character.
     */
    loadGLBModel(url, onStatus) {
      return new Promise((resolve) => {
        if (typeof THREE.GLTFLoader === "undefined") {
          if (onStatus) onStatus("fallback_no_loader");
          resolve(false);
          return;
        }
        if (typeof window !== "undefined" && window.location && window.location.protocol === "file:") {
          if (onStatus) onStatus("fallback_file_protocol");
          resolve(false);
          return;
        }

        const loader = new THREE.GLTFLoader();
        loader.load(
          url || "./assets/investigator.glb",
          (gltf) => {
            const model = gltf.scene;
            model.name = "InvestigatorGLB";

            // Apply tactical investigative suit finish (charcoal coat + muted slate visor)
            model.traverse((node) => {
              if (node.isMesh) {
                node.castShadow = true;
                node.receiveShadow = true;
                if (node.material) {
                  node.material = node.material.clone();
                  const mName = (node.material.name || "").toLowerCase();
                  if (mName.includes("visor")) {
                    node.material.color.setHex(0x1f2937);
                    node.material.emissive = new THREE.Color(0x0c4a6e);
                    node.material.emissiveIntensity = 0.35;
                    node.material.roughness = 0.22;
                    node.material.metalness = 0.75;
                  } else {
                    // Dark investigative uniform tone instead of sci-fi camouflage
                    node.material.color.setHex(0x64748b);
                    node.material.roughness = 0.68;
                    node.material.metalness = 0.25;
                  }
                }
              }
              if (node.isBone && node.name && node.name.toLowerCase().includes("righthand")) {
                this.rightHandBone = node;
              }
            });

            // Attach compact investigative hand scanner onto RightHand bone if found
            if (this.rightHandBone) {
              const scannerMesh = new THREE.Mesh(
                new THREE.BoxGeometry(6, 5, 16),
                new THREE.MeshStandardMaterial({
                  color: 0x0f172a,
                  emissive: 0x38bdf8,
                  emissiveIntensity: 0.45,
                  roughness: 0.3,
                  metalness: 0.6,
                })
              );
              scannerMesh.position.set(2, 8, 2);
              this.rightHandBone.add(scannerMesh);
            }

            // Remove procedural fallback once rigged GLB is ready
            if (this.fallbackRig && this.fallbackRig.rig) {
              this.root.remove(this.fallbackRig.rig);
            }
            this.modelGroup = model;
            this.root.add(model);
            this.isGLBLoaded = true;

            // Configure AnimationMixer & clips (Idle, Walk, Run, TPose)
            this.mixer = new THREE.AnimationMixer(model);
            this.actions = {};
            const clips = gltf.animations || [];
            clips.forEach((clip) => {
              const action = this.mixer.clipAction(clip);
              action.enabled = true;
              this.actions[clip.name] = action;
            });

            const idleAction = this.actions["Idle"] || this.actions[Object.keys(this.actions)[0]];
            if (idleAction) {
              idleAction.reset().setEffectiveTimeScale(1).setEffectiveWeight(1).play();
              this.activeActionName = "Idle";
            }

            if (onStatus) onStatus("loaded_glb");
            resolve(true);
          },
          undefined,
          () => {
            if (onStatus) onStatus("fallback_load_error");
            resolve(false);
          }
        );
      });
    }

    setColliders(colliders) {
      this.colliders = colliders || [];
    }

    setDoorUnlocked(doorTag, unlocked) {
      this.colliders.forEach((c) => {
        if (c.tag === doorTag) {
          c.active = !unlocked;
        }
      });
    }

    _collides(x, z) {
      const r = this.radius;
      // Outer facility perimeter safety clamps
      if (x < -14.2 || x > 14.2 || z < -19.2 || z > 21.2) return true;
      for (let i = 0; i < this.colliders.length; i++) {
        const c = this.colliders[i];
        if (c.active === false) continue;
        if (x + r > c.xmin && x - r < c.xmax && z + r > c.zmin && z - r < c.zmax) {
          return true;
        }
      }
      return false;
    }

    /**
     * Moves investigator horizontally with sliding wall/furniture collision resolution.
     */
    moveBy(dx, dz, dt, isRunning) {
      const dist = Math.hypot(dx, dz);
      this.isRunning = Boolean(isRunning);
      if (dist < 1e-4) {
        this.isMoving = false;
        this._transitionAnimation("Idle", 0.22);
        return false;
      }

      this.isMoving = true;
      const speed = this.isRunning ? this.runSpeed : this.walkSpeed;
      const step = Math.min(dist, speed * dt);
      const nx = (dx / dist) * step;
      const nz = (dz / dist) * step;

      let moved = false;
      if (!this._collides(this.position.x + nx, this.position.z)) {
        this.position.x += nx;
        moved = true;
      }
      if (!this._collides(this.position.x, this.position.z + nz)) {
        this.position.z += nz;
        moved = true;
      }

      // Compute desired heading so character faces direction of travel (-Z local forward)
      const targetYaw = Math.atan2(-nx, -nz);
      const diff = wrapAngle(targetYaw - this.facingYaw);
      this.facingYaw = wrapAngle(this.facingYaw + diff * Math.min(1.0, dt * 11.0));

      this._transitionAnimation(this.isRunning ? "Run" : "Walk", 0.18);
      return moved;
    }

    isBlockedAt(x, z) {
      return this._collides(x, z);
    }

    /**
     * Moves investigator toward (dx, dz), steering smoothly around blocking furniture
     * or doorway jambs instead of stalling when head-on axis-aligned movement is blocked.
     */
    steerMove(dx, dz, dt, isRunning) {
      const dist = Math.hypot(dx, dz);
      if (dist < 1e-4) {
        return this.moveBy(0, 0, dt, isRunning);
      }

      const prevX = this.position.x;
      const prevZ = this.position.z;
      this.moveBy(dx, dz, dt, isRunning);
      const progressed = Math.hypot(this.position.x - prevX, this.position.z - prevZ);
      if (progressed > 0.005) {
        this._steerSign = 0;
        return true;
      }

      // Direct line is blocked — probe angled bypass directions
      const baseAngle = Math.atan2(dz, dx);
      const pref = this._steerSign || 1;
      const offsetsDeg = [45 * pref, -45 * pref, 75 * pref, -75 * pref, 95 * pref, -95 * pref, 120 * pref, -120 * pref];
      for (let i = 0; i < offsetsDeg.length; i++) {
        const rad = baseAngle + (offsetsDeg[i] * Math.PI) / 180;
        const sx = Math.cos(rad) * dist;
        const sz = Math.sin(rad) * dist;
        const probeX = prevX + Math.cos(rad) * Math.min(0.45, Math.max(0.25, dt * this.walkSpeed));
        const probeZ = prevZ + Math.sin(rad) * Math.min(0.45, Math.max(0.25, dt * this.walkSpeed));
        if (!this._collides(probeX, probeZ)) {
          this._steerSign = offsetsDeg[i] >= 0 ? 1 : -1;
          return this.moveBy(sx, sz, dt, isRunning);
        }
      }
      return false;
    }

    teleportTo(x, z) {
      this.position.set(x, 0.0, z);
      this.root.position.copy(this.position);
      this.isMoving = false;
      this._transitionAnimation("Idle", 0.15);
    }

    triggerInteract() {
      this.interactTimer = 0.55;
    }

    _transitionAnimation(nextName, duration) {
      if (!this.isGLBLoaded || !this.actions[nextName] || this.activeActionName === nextName) {
        return;
      }
      const prev = this.actions[this.activeActionName];
      const next = this.actions[nextName];
      this.activeActionName = nextName;
      next.reset();
      next.setEffectiveTimeScale(1);
      next.setEffectiveWeight(1);
      next.play();
      if (prev && prev !== next) {
        prev.crossFadeTo(next, duration || 0.2, true);
      }
    }

    setCameraMode(mode) {
      this.cameraMode = mode;
      const showBody = mode !== "1st";
      this.root.visible = showBody;
    }

    /**
     * Updates skeletal animation, fallback gait, and smooth camera following.
     */
    update(dt, camera) {
      // Sync root transform
      this.root.position.copy(this.position);
      this.root.rotation.y = this.cameraMode === "1st" ? this.cameraYaw : this.facingYaw;

      if (this.interactTimer > 0) {
        this.interactTimer = Math.max(0, this.interactTimer - dt);
      }

      // Update skeletal GLB AnimationMixer
      if (this.mixer) {
        this.mixer.update(dt);
        if (this.rightHandBone && this.interactTimer > 0) {
          this.rightHandBone.rotation.x = -0.65 * Math.sin((this.interactTimer / 0.55) * Math.PI);
        }
      } else if (this.fallbackRig) {
        // Animate procedural fallback rig smoothly
        const rig = this.fallbackRig;
        if (this.isMoving) {
          rig.phase += dt * (this.isRunning ? 12.5 : 8.5);
          const swing = Math.sin(rig.phase) * (this.isRunning ? 0.65 : 0.45);
          rig.leftLeg.rotation.x = swing;
          rig.rightLeg.rotation.x = -swing;
          rig.leftArm.rotation.x = -swing * 0.7;
          rig.rightArm.rotation.x = this.interactTimer > 0 ? -0.9 : swing * 0.7;
        } else {
          rig.phase += dt * 1.8;
          rig.leftLeg.rotation.x *= 0.85;
          rig.rightLeg.rotation.x *= 0.85;
          rig.leftArm.rotation.x *= 0.85;
          rig.rightArm.rotation.x = this.interactTimer > 0 ? -0.9 : Math.sin(rig.phase) * 0.03;
        }
      }

      // Camera Following & Perspective Alignment
      if (!camera) return;
      const forwardX = -Math.sin(this.cameraYaw);
      const forwardZ = -Math.cos(this.cameraYaw);

      if (this.cameraMode === "1st") {
        const eyePos = new THREE.Vector3(this.position.x, 1.66, this.position.z);
        const lookAt = new THREE.Vector3(
          this.position.x + forwardX * 6.0,
          1.66 + Math.sin(this.cameraPitch) * 6.0,
          this.position.z + forwardZ * 6.0
        );
        camera.position.lerp(eyePos, Math.min(1, dt * 18));
        camera.lookAt(lookAt);
      } else if (this.cameraMode === "top") {
        const targetPos = new THREE.Vector3(this.position.x, 19.5, this.position.z - 2.2);
        const targetLook = new THREE.Vector3(this.position.x, 0.6, this.position.z);
        this.cameraPosSmooth.lerp(targetPos, Math.min(1, dt * 7));
        this.cameraLookSmooth.lerp(targetLook, Math.min(1, dt * 9));
        camera.position.copy(this.cameraPosSmooth);
        camera.lookAt(this.cameraLookSmooth);
      } else {
        // 3rd Person smooth trailing camera with pitch & wall clamping
        const trailDist = 3.2;
        const pitchHeight = 2.15 - Math.sin(this.cameraPitch) * 1.9;
        let camX = this.position.x - forwardX * trailDist;
        let camZ = this.position.z - forwardZ * trailDist;
        const camY = Math.max(1.15, Math.min(3.75, pitchHeight));

        // Clamp camera safely inside outer ceiling/wall boundaries
        camX = Math.max(-11.8, Math.min(11.8, camX));
        camZ = Math.max(-20.1, Math.min(21.8, camZ));

        const targetPos = new THREE.Vector3(camX, camY, camZ);
        const targetLook = new THREE.Vector3(
          this.position.x + forwardX * 2.0,
          1.38 + Math.sin(this.cameraPitch) * 1.1,
          this.position.z + forwardZ * 2.0
        );

        this.cameraPosSmooth.lerp(targetPos, Math.min(1, dt * 9));
        this.cameraLookSmooth.lerp(targetLook, Math.min(1, dt * 11));
        camera.position.copy(this.cameraPosSmooth);
        camera.lookAt(this.cameraLookSmooth);
      }
    }
  }

  global.SpatialInvestigator3D = { InvestigatorController };
})(window);
