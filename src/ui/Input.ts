// Keyboard + mouse (pointer lock) + touch controls.

export class Input {
  keys = new Set<string>();
  mouseDown = false;
  /** Accumulated look delta since last read (radians). */
  lookX = 0;
  lookY = 0;
  wheel = 0;
  pressed = new Set<string>();
  locked = false;
  touch = false;
  // Touch state
  moveX = 0;
  moveY = 0;
  touchFire = false;
  private stickId: number | null = null;
  private stickOrigin = [0, 0];
  private lookId: number | null = null;
  private lookLast = [0, 0];
  sensitivity = 0.0022;

  constructor(canvas: HTMLCanvasElement) {
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      this.keys.add(e.code);
      this.pressed.add(e.code);
      if (['Space', 'ArrowUp', 'ArrowDown', 'Tab'].includes(e.code)) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => {
      this.keys.clear();
      this.mouseDown = false;
    });
    canvas.addEventListener('click', () => {
      if (!this.touch && !this.locked) canvas.requestPointerLock?.();
    });
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === canvas;
      if (!this.locked) this.mouseDown = false;
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.locked) return;
      this.lookX += e.movementX * this.sensitivity;
      this.lookY += e.movementY * this.sensitivity;
    });
    window.addEventListener('mousedown', (e) => {
      if (!this.locked) return;
      if (e.button === 0) this.mouseDown = true;
      if (e.button === 2) this.pressed.add('Mouse2');
    });
    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) this.mouseDown = false;
    });
    window.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('wheel', (e) => {
      if (this.locked) this.wheel += Math.sign(e.deltaY);
    }, { passive: true });

    canvas.addEventListener('touchstart', (e) => this.onTouchStart(e), { passive: false });
    canvas.addEventListener('touchmove', (e) => this.onTouchMove(e), { passive: false });
    canvas.addEventListener('touchend', (e) => this.onTouchEnd(e), { passive: false });
    canvas.addEventListener('touchcancel', (e) => this.onTouchEnd(e), { passive: false });
  }

  private onTouchStart(e: TouchEvent) {
    e.preventDefault();
    this.touch = true;
    for (const t of Array.from(e.changedTouches)) {
      if (t.clientX < window.innerWidth * 0.4 && this.stickId === null) {
        this.stickId = t.identifier;
        this.stickOrigin = [t.clientX, t.clientY];
      } else if (this.lookId === null) {
        this.lookId = t.identifier;
        this.lookLast = [t.clientX, t.clientY];
      }
    }
  }

  private onTouchMove(e: TouchEvent) {
    e.preventDefault();
    for (const t of Array.from(e.changedTouches)) {
      if (t.identifier === this.stickId) {
        const dx = (t.clientX - this.stickOrigin[0]) / 60;
        const dy = (t.clientY - this.stickOrigin[1]) / 60;
        const l = Math.hypot(dx, dy);
        const k = l > 1 ? 1 / l : 1;
        this.moveX = dx * k;
        this.moveY = -dy * k;
      } else if (t.identifier === this.lookId) {
        this.lookX += (t.clientX - this.lookLast[0]) * 0.005;
        this.lookY += (t.clientY - this.lookLast[1]) * 0.005;
        this.lookLast = [t.clientX, t.clientY];
      }
    }
  }

  private onTouchEnd(e: TouchEvent) {
    for (const t of Array.from(e.changedTouches)) {
      if (t.identifier === this.stickId) {
        this.stickId = null;
        this.moveX = this.moveY = 0;
      } else if (t.identifier === this.lookId) {
        this.lookId = null;
      }
    }
  }

  /** Was this key pressed since the last call to endFrame()? */
  hit(code: string): boolean {
    return this.pressed.has(code);
  }

  down(code: string): boolean {
    return this.keys.has(code);
  }

  endFrame() {
    this.pressed.clear();
    this.lookX = 0;
    this.lookY = 0;
    this.wheel = 0;
  }
}
