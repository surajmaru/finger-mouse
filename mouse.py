import cv2
import mediapipe as mp
import pyautogui
import math
import time
from collections import deque

# ============================================================
#  Air Mouse v3 — Production Ready (fixed)
#  - One Euro Filter (primary smoother)
#  - Kalman Filter (snap recovery only, conservative)
#  - 3D pinch distance (rotation-invariant)
#  - Gentle acceleration curve (no crazy amplification)
#  - Boundary deceleration
#  - Hand-loss hold
#  - Click flash feedback
#  - Live trackbar tuning
# ============================================================

class OneEuroFilter:
    def __init__(self, freq=60, mincutoff=0.01, beta=0.007, dcutoff=1.0):
        self.freq      = freq
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self.x_prev    = None
        self.dx_prev   = 0.0
        self.last_t    = None

    def _sf(self, cutoff):
        te = 1.0 / self.freq
        r  = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def filter(self, x, t):
        if self.last_t is None:
            self.last_t = t
            self.x_prev = x
            return x
        dt = min(t - self.last_t, 0.1)
        if dt <= 0:
            return self.x_prev
        self.freq    = 1.0 / dt
        self.last_t  = t
        dx           = (x - self.x_prev) * self.freq
        alpha_d      = self._sf(self.dcutoff)
        dx_hat       = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat
        cutoff       = self.mincutoff + self.beta * abs(dx_hat)
        alpha        = self._sf(cutoff)
        x_hat        = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev  = x_hat
        return x_hat


class KalmanFilter1D:
    """Conservative Kalman — only smooths big snaps, not micro-movement."""
    def __init__(self, process_noise=1.0, measurement_noise=25.0):
        self.q    = process_noise
        self.r    = measurement_noise
        self.x    = 0.0
        self.p    = 1.0
        self.init = False

    def filter(self, z):
        if not self.init:
            self.x    = z
            self.init = True
            return z
        p_pred = self.p + self.q
        k      = p_pred / (p_pred + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * p_pred
        return self.x


def apply_accel(dx, dy, threshold=8.0, boost=1.6):
    """
    Gentle acceleration:
    - Below threshold pixels/frame → no boost (precise targeting)
    - Above threshold → linearly ramp up to boost (max 2x, not 4x)
    """
    speed = math.hypot(dx, dy)
    if speed < 1e-6 or speed <= threshold:
        return dx, dy
    factor = 1.0 + (boost - 1.0) * min((speed - threshold) / threshold, 1.0)
    factor = min(factor, boost)
    return dx * factor, dy * factor


def boundary_decel(x, y, sw, sh, margin=60):
    def edge(val, lo, hi, m):
        if val < lo + m:
            return max(0.3, (val - lo) / m)
        if val > hi - m:
            return max(0.3, (hi - val) / m)
        return 1.0
    return min(edge(x, 0, sw, margin), edge(y, 0, sh, margin))


def dist3d(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def remap(val, in_lo, in_hi, out_lo, out_hi):
    val = max(in_lo, min(in_hi, val))
    return out_lo + (val - in_lo) / (in_hi - in_lo) * (out_hi - out_lo)


# ----------------------------
# Setup
# ----------------------------
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.80,
    min_tracking_confidence=0.80,
    model_complexity=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          60)

screen_w, screen_h = pyautogui.size()

oef_x = OneEuroFilter(mincutoff=0.01,  beta=0.007)
oef_y = OneEuroFilter(mincutoff=0.012, beta=0.009)

# Kalman is now conservative (high measurement_noise = trusts filter more than raw)
kal_x = KalmanFilter1D(process_noise=1.0, measurement_noise=25.0)
kal_y = KalmanFilter1D(process_noise=1.0, measurement_noise=25.0)

SMOOTH_SIZE = 4
q_x = deque(maxlen=SMOOTH_SIZE)
q_y = deque(maxlen=SMOOTH_SIZE)

# ----------------------------
# Live tuning window
# Sane defaults — these work for most people out of the box
# ----------------------------
TUNE_WIN = "Tuning"
cv2.namedWindow(TUNE_WIN)
cv2.resizeWindow(TUNE_WIN, 420, 180)
# Gain: trackbar 10 = 1.0x, 20 = 2.0x  (default 18 = 1.8x — safe start)
cv2.createTrackbar("Gain  (x10)",        TUNE_WIN, 18,  50,  lambda v: None)
# Dead zone: trackbar value = pixels  (default 25 = 2.5px)
cv2.createTrackbar("Dead zone (x10)",    TUNE_WIN, 25,  100, lambda v: None)
# Pinch close: trackbar value / 100  (default 18 = 0.18)
cv2.createTrackbar("Pinch close (x100)", TUNE_WIN, 18,  60,  lambda v: None)
# Pinch open: trackbar value / 100   (default 28 = 0.28)
cv2.createTrackbar("Pinch open  (x100)", TUNE_WIN, 28,  80,  lambda v: None)

# ----------------------------
# State
# ----------------------------
cursor_x, cursor_y = float(screen_w // 2), float(screen_h // 2)

frozen            = False
pinch_armed       = True
last_click_t      = 0.0
CLICK_COOLDOWN    = 0.4

HAND_HOLD_FRAMES  = 8
no_hand_frames    = 0

click_flash_until = 0.0
FLASH_DURATION    = 0.18

MARGIN = 0.10

print("Air Mouse v3 — run and drag sliders in the Tuning window")
print("ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    t = time.time()

    # Read trackbars
    GAIN              = max(cv2.getTrackbarPos("Gain  (x10)",        TUNE_WIN) / 10.0, 0.5)
    DEAD_ZONE         = cv2.getTrackbarPos("Dead zone (x10)",    TUNE_WIN) / 10.0
    PINCH_CLOSE_RATIO = cv2.getTrackbarPos("Pinch close (x100)", TUNE_WIN) / 100.0
    PINCH_OPEN_RATIO  = cv2.getTrackbarPos("Pinch open  (x100)", TUNE_WIN) / 100.0
    PINCH_CLOSE_RATIO = max(PINCH_CLOSE_RATIO, 0.05)
    PINCH_OPEN_RATIO  = max(PINCH_OPEN_RATIO, PINCH_CLOSE_RATIO + 0.05)

    if results.multi_hand_landmarks:
        no_hand_frames = 0
        hand = results.multi_hand_landmarks[0]
        lm   = hand.landmark

        knuckle   = lm[5]
        index_tip = lm[8]
        thumb_tip = lm[4]
        wrist     = lm[0]
        mid_mcp   = lm[9]

        hand_size   = max(dist2d(wrist, mid_mcp), 0.01)
        pinch_d     = dist3d(index_tip, thumb_tip)
        pinch_ratio = pinch_d / hand_size

        is_pinched = pinch_ratio < PINCH_CLOSE_RATIO
        is_open    = pinch_ratio > PINCH_OPEN_RATIO

        # State machine
        if is_open:
            frozen      = False
            pinch_armed = True
        elif is_pinched and pinch_armed:
            if (t - last_click_t) > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_t      = t
                click_flash_until = t + FLASH_DURATION
            frozen      = True
            pinch_armed = False
        elif is_pinched and not pinch_armed:
            frozen = True
        else:
            # closing zone — freeze preemptively
            frozen = True

        # Cursor movement
        if not frozen:
            raw_x = remap(knuckle.x, MARGIN, 1.0 - MARGIN, 0.0, 1.0)
            raw_y = remap(knuckle.y, MARGIN, 1.0 - MARGIN, 0.0, 1.0)

            target_x = (raw_x - 0.5) * GAIN * screen_w + screen_w / 2
            target_y = (raw_y - 0.5) * GAIN * screen_h + screen_h / 2
            target_x = max(0.0, min(float(screen_w - 1), target_x))
            target_y = max(0.0, min(float(screen_h - 1), target_y))

            # One Euro
            fx_val = oef_x.filter(target_x, t)
            fy_val = oef_y.filter(target_y, t)

            # Kalman (conservative — only kills hard snaps)
            kx_val = kal_x.filter(fx_val)
            ky_val = kal_y.filter(fy_val)

            # Moving average
            q_x.append(kx_val)
            q_y.append(ky_val)
            smooth_x = sum(q_x) / len(q_x)
            smooth_y = sum(q_y) / len(q_y)

            # Delta-based acceleration (gentle, max 1.6x boost)
            dx = smooth_x - cursor_x
            dy = smooth_y - cursor_y
            dx, dy = apply_accel(dx, dy, threshold=8.0, boost=1.6)

            new_x = max(0.0, min(float(screen_w - 1), cursor_x + dx))
            new_y = max(0.0, min(float(screen_h - 1), cursor_y + dy))

            # Boundary decel
            bd    = boundary_decel(new_x, new_y, screen_w, screen_h)
            new_x = cursor_x + (new_x - cursor_x) * bd
            new_y = cursor_y + (new_y - cursor_y) * bd

            if abs(new_x - cursor_x) > DEAD_ZONE or abs(new_y - cursor_y) > DEAD_ZONE:
                cursor_x = new_x
                cursor_y = new_y

        pyautogui.moveTo(int(cursor_x), int(cursor_y))

        # Overlay
        kx_px = int(knuckle.x * w)
        ky_px = int(knuckle.y * h)
        it_px = (int(index_tip.x * w), int(index_tip.y * h))
        tt_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))

        track_col = (100, 100, 100) if frozen else (0, 200, 255)
        line_col  = (0, 0, 220)     if is_pinched else (0, 220, 0)

        cv2.circle(frame, (kx_px, ky_px), 12, track_col, -1)
        cv2.circle(frame, (kx_px, ky_px), 12, (255,255,255), 1)
        cv2.circle(frame, it_px, 8, line_col, -1)
        cv2.circle(frame, tt_px, 8, (255,180,0), -1)
        cv2.line(frame, it_px, tt_px, line_col, 2)

        bar     = int(remap(pinch_ratio, 0, 0.5, 0, 200))
        bar_col = (0,0,220) if is_pinched else (0,180,0)
        cv2.rectangle(frame, (10, h-28), (10+bar, h-10), bar_col, -1)
        cv2.rectangle(frame, (10, h-28), (210,    h-10), (180,180,180), 1)
        cpx = 10 + int(remap(PINCH_CLOSE_RATIO, 0, 0.5, 0, 200))
        opx = 10 + int(remap(PINCH_OPEN_RATIO,  0, 0.5, 0, 200))
        cv2.line(frame, (cpx, h-32), (cpx, h-8), (0,0,255), 2)
        cv2.line(frame, (opx, h-32), (opx, h-8), (0,255,0), 2)

        state = "FROZEN-clicked" if (frozen and not pinch_armed) else \
                "FROZEN-closing" if frozen else "TRACKING"
        cv2.putText(frame, f"{state}  ratio={pinch_ratio:.2f}",
                    (10, h-36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(frame, f"cursor ({int(cursor_x)},{int(cursor_y)})  gain={GAIN:.1f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    else:
        no_hand_frames += 1
        if no_hand_frames > HAND_HOLD_FRAMES:
            cv2.putText(frame, "No hand detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,100,255), 2)

    # Click flash
    if t < click_flash_until:
        cx, cy  = w // 2, h // 2
        alpha   = (click_flash_until - t) / FLASH_DURATION
        radius  = int(30 + (1 - alpha) * 20)
        thick   = max(1, int(4 * alpha))
        cv2.circle(frame, (cx, cy), radius, (0, 255, 80), thick)
        cv2.putText(frame, "CLICK", (cx-22, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,80), 2)

    cv2.imshow("Air Mouse v3", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()