import numpy as np
import time
import threading
import os
import math
import cv2
import queue
import logging
from logging.handlers import RotatingFileHandler

from mouse import Mouse, is_button_pressed
from capture import get_camera
from detection import load_model, perform_detection, get_stats
from config import config
from windmouse_smooth import smooth_aimer
from jitter import JitterController

# === ИМПОРТ DETECTION_V2 v2.12 ===
from detection_v2 import (
    DeadzoneFilter,
    DetectorConfig,
    Category,
    detect,
    draw_debug_overlay,
    TargetLock,
    LockConfig,
    LockStatus,
    EgoTracker,
    TargetLifecycle,
    log_startup_config,
    ButtonDebounce
)

# === Логирование ===
def _setup_logging():
    level_name = os.environ.get("AIMBOT_LOG_LEVEL", getattr(config, "log_level", "WARNING")).upper()
    level = getattr(logging, level_name, logging.WARNING)
    fmt = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s | %(message)s', datefmt='%H:%M:%S')
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        os.makedirs("logs", exist_ok=True)
        fh = RotatingFileHandler("logs/aimbot.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception as e:
        print(f"[WARN] Не удалось создать файл логов: {e}")

_setup_logging()
logger = logging.getLogger("aimbot")

# === Глобальное состояние для GUI ===
fps = 0.0
_aimbot_running = False
makcu = None


class Aimbot:
    def __init__(self):
        self.stop_event = threading.Event()
        self._running = False
        self.fps = 0.0

        self.capture_thread = None
        self.smooth_thread = None
        self.aim_thread = None

        self.frame_queue = queue.Queue(maxsize=1)
        self.smooth_move_queue = queue.Queue(maxsize=10)

        self.makcu = None
        self._owns_makcu = False
        # УБРАНО: self._makcu_lock больше не нужен, так как только один поток обращается к makcu
        
        self.jitter_controller = None
        self.model = None
        self.class_names = {}

        # EMA состояние
        self.ema_target_x = None
        self.ema_target_y = None
        self.jump_threshold_px = 250.0

        # Диагностика
        self._had_target = False
        self._last_best_center = None
        self._frame_ms_ema = 0.0
        self._slow_frames = 0
        self._lock_waits = 0
        self._ndi_no_frame_streak = 0
        self._last_summary_ts = time.perf_counter()
        self._last_target_ts = 0.0
        self._last_dist = 999.0
        self._jitter_deactivate_ts = 0.0
        self._last_frame_ts = 0.0
        self._frame_dt = 0.014
        
        # Для компенсации ego_shift
        self._last_aim_dx = 0.0
        self._last_aim_dy = 0.0

        # 4. ОПТИМИЗАЦИЯ: Кэширование distance_multiplier (вычисляется 1 раз)
        sens = getattr(config, 'in_game_sens', 1.5)
        self._distance_multiplier = 1.07437623 * math.pow(sens, -0.9936827126)

        # 3. ОПТИМИЗАЦИЯ: Флаги для безопасной инициализации OpenCV вне цикла
        self._debug_window_initialized = False
        self._debug_win_name = "AI Debug"

        # === ИНТЕГРАЦИЯ DETECTION_V2 ===
        self.deadzone = DeadzoneFilter(
            enter=getattr(config, 'deadzone_enter', 0.8),
            exit_ratio=0.6,
            radial=True,
            rescale=True
        )

        self.button_debounce = ButtonDebounce(
            release_frames=getattr(config, 'button_release_frames', 1),
            always_on=getattr(config, 'always_on_aim', False)
        )

        self.ego_tracker = EgoTracker(
            px_per_unit=getattr(config, 'ego_px_per_unit', 0.7),
            jitter_offset=self.jitter_controller.offset_at if self.jitter_controller else None,
            latency_s=getattr(config, 'ego_latency_s', 0.03),
            aim_latency_s=getattr(config, 'aim_latency_s', 0.03),
            aim_smear_s=getattr(config, 'aim_smear_s', 0.04)
        )

        lock_cfg = LockConfig.recommended(
            lost_frames=getattr(config, 'lock_lost_frames', 15),
            match_radius=getattr(config, 'lock_match_radius', 0.1),
            acquire_frames=getattr(config, 'lock_acquire_frames', 2),
            acquire_conf=getattr(config, 'lock_acquire_conf', 0.35),
            coast_radius_max=getattr(config, 'lock_coast_radius_max', 1.0),
            hard_lock=getattr(config, 'lock_hard_lock', True),
            radius_heights=getattr(config, 'lock_radius_heights', 2.0),
            size_ratio_max=getattr(config, 'lock_size_ratio_max', 2.0),
            neighbor_guard=getattr(config, 'lock_neighbor_guard', 0.5),
            relock_after_lost=getattr(config, 'lock_relock_after_lost', False),
            size_free_heights=getattr(config, 'lock_size_free_heights', 0.5),
        )
        self.target_lock = TargetLock(lock_cfg)

        self.lifecycle = TargetLifecycle(
            lock=self.target_lock,
            debounce=self.button_debounce,
            priority_only=True,
            short_hold_frames=getattr(config, 'short_hold_frames', 3),
            logger=logger
        )

        priority_labels = set()
        if config.custom_player_label:
            priority_labels.add(config.custom_player_label.lower())
        if config.custom_head_label:
            priority_labels.add(config.custom_head_label.lower())

        self.detector_config = DetectorConfig(
            conf_threshold=getattr(config, 'conf', 0.20),
            poi_offset=getattr(config, 'player_poi_offset', 0.2),
            use_tracking=getattr(config, 'use_tracking', False),
            priority_labels=frozenset(priority_labels),
            aim_margin_px=getattr(config, 'aim_margin_px', 4.0),
            ref_center=None
        )

    def _now_ms(self):
        return time.perf_counter() * 1000.0

    # 2. ОПТИМИЗАЦИЯ: Убраны self._makcu_lock. Этот метод теперь вызывается ТОЛЬКО из smooth_movement_loop
    def _locked_move(self, dx, dy):
        if self.makcu is not None:
            self.makcu.move(dx, dy)

    def _locked_click(self):
        if self.makcu is not None:
            self.makcu.click()

    def smooth_movement_loop(self):
        logger.info("Smooth movement thread started (Sole owner of makcu)")
        while not self.stop_event.is_set():
            try:
                move_data = self.smooth_move_queue.get(timeout=0.05)
                dx, dy, delay = move_data
                
                if self.makcu is not None:
                    self._locked_move(dx, dy)
                
                if delay > 0:
                    time.sleep(delay)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Smooth movement failed: {e}")
                time.sleep(0.01)
        logger.info("Smooth movement thread stopped")

    def capture_loop(self):
        camera, _ = get_camera()
        last_selected = None
        last_sources_update = 0.0

        while not self.stop_event.is_set():
            try:
                now = time.perf_counter()
                if now - last_sources_update > 3.0:
                    try:
                        config.ndi_sources = camera.list_sources()
                        last_sources_update = now
                    except Exception:
                        config.ndi_sources = []

                if config.capturer_mode.lower() == "ndi":
                    desired = config.ndi_selected_source
                    if isinstance(desired, str) and desired in config.ndi_sources:
                        if (desired != last_selected) or not camera.connected:
                            camera.select_source(desired)
                            last_selected = desired
                            logger.info("NDI source selected: %s", desired)

                image = camera.get_latest_frame()
                if image is not None:
                    if self._ndi_no_frame_streak:
                        logger.info("NDI frames resumed after %d empty polls", self._ndi_no_frame_streak)
                    self._ndi_no_frame_streak = 0
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self.frame_queue.put_nowait(image)
                    except queue.Full:
                        pass
                else:
                    time.sleep(0.001)
                    self._ndi_no_frame_streak += 1
                    if self._ndi_no_frame_streak == 60:
                        logger.warning("NDI: нет кадров от '%s' (60 опросов)", config.ndi_selected_source)
            except Exception as e:
                logger.error(f"Capture loop failed: {e}")
                time.sleep(1)
        try:
            camera.stop()
        except Exception:
            pass
        logger.info("Capture loop stopped")

    # 1. и 2. ОПТИМИЗАЦИЯ: Полностью неблокирующая отправка в очередь. Никаких lock.
    def _apply_movement_mode(self, dx, dy, frame_count, best_target=None):
        final_dx, final_dy = 0.0, 0.0
        
        if config.mode == "normal":
            dt = self._frame_dt
            dist_px = math.hypot(dx, dy)
            
            if dist_px > 80: base_gain = 0.55
            elif dist_px > 40: base_gain = 0.45
            elif dist_px > 15: base_gain = 0.35
            else: base_gain = 0.25

            tau_x = 0.010 / max(config.normal_x_speed, 0.01)
            tau_y = 0.010 / max(config.normal_y_speed, 0.01)
            gain_x = min(1.0 - math.exp(-dt / tau_x), base_gain)
            gain_y = min(1.0 - math.exp(-dt / tau_y), base_gain)
            
            dx *= gain_x
            dy *= gain_y

            max_step = max(6.0, 2600.0 * dt)
            step = math.hypot(dx, dy)
            if step > max_step:
                k = max_step / step
                dx *= k
                dy *= k

            final_dx, final_dy = self.deadzone.apply(dx, dy)

        elif config.mode == "bezier":
            # Для упрощения конвейера bezier тоже считает финальные координаты и отправляет в очередь
            # (Внутренняя логика bezier может быть адаптирована под этот паттерн)
            final_dx, final_dy = dx, dy 
        elif config.mode == "silent":
            final_dx, final_dy = dx, dy
        elif config.mode == "smooth":
            # smooth уже работает через очередь, но для унификации:
            path = smooth_aimer.calculate_smooth_path(dx, dy, config)
            for move_dx, move_dy, delay in path:
                try:
                    self.smooth_move_queue.put_nowait((move_dx, move_dy, delay))
                except queue.Full:
                    break
            return # Если smooth, мы уже заполнили очередь, выходим

        # Отправка в очередь БЕЗ блокировок (Zero-lock pipeline)
        if final_dx != 0.0 or final_dy != 0.0:
            try:
                self.smooth_move_queue.put_nowait((final_dx, final_dy, 0.0))
            except queue.Full:
                pass # Пропускаем кадр, если очередь переполнена, чтобы не тормозить инференс

        # Сохраняем для ego_shift
        self._last_aim_dx = final_dx
        self._last_aim_dy = final_dy

    def _process_jitter(self, button_held: bool):
        if not getattr(config, 'jitter_enabled', False) or not self.jitter_controller:
            return

        trigger_btn = int(getattr(config, 'jitter_trigger_button', 0))
        require_ads = getattr(config, 'jitter_require_ads', False)
        is_trigger_pressed = is_button_pressed(trigger_btn)
        is_rmb_pressed = is_button_pressed(1)

        is_activated = button_held and is_trigger_pressed and (not require_ads or is_rmb_pressed)
        now = time.perf_counter()

        if is_activated:
            self._jitter_deactivate_ts = 0.0
            if not self.jitter_controller.is_active:
                self.jitter_controller.update_params()
                if not self.jitter_controller.start():
                    pass 
                else:
                    logger.info("JITTER START | pattern=%s amp=%d/%d",
                                self.jitter_controller.pattern, self.jitter_controller.amp_x, self.jitter_controller.amp_y)
        else:
            if self._jitter_deactivate_ts == 0.0:
                self._jitter_deactivate_ts = now

            if self.jitter_controller.is_active and (now - self._jitter_deactivate_ts) > 0.20:
                self.jitter_controller.request_stop()
                logger.info("JITTER STOP")

    def detection_and_aim_loop(self):
        global fps
        self.model, self.class_names = load_model(config.model_path)
        if self.model is not None:
            logger.info("Warming up model (10 frames)...")
            dummy_image = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
            for _ in range(10):
                perform_detection(self.model, dummy_image)
            logger.info("Warmup complete")

        frame_count = 0
        start_time = time.perf_counter()

        while not self.stop_event.is_set():
            try:
                image = self.frame_queue.get(timeout=1)
            except queue.Empty:
                logger.warning("Frame queue is empty. Capture thread may have stalled.")
                continue

            if self.model is None:
                time.sleep(0.1)
                frame_count += 1
                continue

            t_frame = time.perf_counter()
            if self._last_frame_ts:
                self._frame_dt = min(max(t_frame - self._last_frame_ts, 0.001), 0.05)
            self._last_frame_ts = t_frame

            if config.capturer_mode.lower() == "mss":
                ref_cx, ref_cy = config.region_size / 2.0, config.region_size / 2.0
                region_left = (config.screen_width - config.region_size) // 2
                region_top = (config.screen_height - config.region_size) // 2
                crosshair_x, crosshair_y = config.screen_width // 2, config.screen_height // 2
            else:
                ref_cx, ref_cy = config.ndi_width / 2.0, config.ndi_height / 2.0
                region_left = (config.main_pc_width - config.ndi_width) // 2
                region_top = (config.main_pc_height - config.ndi_height) // 2
                crosshair_x, crosshair_y = config.main_pc_width // 2, config.main_pc_height // 2

            current_config = DetectorConfig(
                conf_threshold=self.detector_config.conf_threshold,
                poi_offset=self.detector_config.poi_offset,
                use_tracking=self.detector_config.use_tracking,
                priority_labels=self.detector_config.priority_labels,
                aim_margin_px=self.detector_config.aim_margin_px,
                ref_center=(ref_cx, ref_cy)
            )
            
            debug_img = image.copy() if config.show_debug_window else None
            result = detect(self.model, image, self.class_names, current_config)
            
            if debug_img is not None:
                draw_debug_overlay(debug_img, result)

            raw_button_held = is_button_pressed(config.selected_mouse_button)
            
            ego_shift = self.ego_tracker.update(
                aim_move=(self._last_aim_dx, self._last_aim_dy),
                t=t_frame
            )
            
            life_result = self.lifecycle.update(
                result.entities, image.shape[:2], raw_button_held, 
                ego_shift=ego_shift, t=t_frame
            )

            best_target = life_result.target
            lock_status = life_result.status
            button_held = life_result.held

            # Блок калибровки (оставлен для финальной сверки)
            if lock_status == LockStatus.TRACKING and best_target is not None and self._last_best_center is not None:
                dx = best_target.poi[0] - self._last_best_center[0]
                dy = best_target.poi[1] - self._last_best_center[1]
                logger.info(f"EGO | dx={dx:.1f} dy={dy:.1f} | mx={self._last_aim_dx:.2f} my={self._last_aim_dy:.2f} | dt={self._frame_dt*1000:.1f}ms")

            if best_target:
                cx, cy = best_target.poi
                self._last_dist = best_target.distance_px
                self._last_target_ts = t_frame
                
                if lock_status == LockStatus.TRACKING:
                    self._last_best_center = (cx, cy)
                    self._had_target = True
                elif lock_status == LockStatus.COASTING:
                    pass 
            else:
                if self._had_target:
                    pass
                self._had_target = False
                self._last_best_center = None
                
                if t_frame - self._last_target_ts > 0.20:
                    smooth_aimer.reset_fatigue()
                    self.ema_target_x = None
                    self.ema_target_y = None
                    self._last_dist = 999.0
                    self.deadzone.reset()

            if best_target:
                tx, ty = best_target.poi
                box_height = best_target.height
                
                if self.ema_target_x is None:
                    self.ema_target_x, self.ema_target_y = tx, ty
                else:
                    jump_dist = math.hypot(tx - self.ema_target_x, ty - self.ema_target_y)
                    
                    # 3. ОПТИМИЗАЦИЯ: Плавная EMA вместо ступенчатого if/else
                    max_dist = 20.0
                    min_alpha, max_alpha = 0.15, 0.85
                    raw_alpha = min_alpha + (max_alpha - min_alpha) * min(jump_dist / max_dist, 1.0)
                    
                    if box_height < 60 and jump_dist > 2.0:
                        raw_alpha = max(raw_alpha, 0.85)
                        
                    alpha = max(min_alpha, min(max_alpha, raw_alpha))
                    
                    self.ema_target_x = alpha * tx + (1 - alpha) * self.ema_target_x
                    self.ema_target_y = alpha * ty + (1 - alpha) * self.ema_target_y

                target_screen_x = region_left + self.ema_target_x
                target_screen_y = region_top + self.ema_target_y
                
                # 4. ОПТИМИЗАЦИЯ: Используем закэшированный множитель
                raw_dx = (target_screen_x - crosshair_x) * self._distance_multiplier
                raw_dy = (target_screen_y - crosshair_y) * self._distance_multiplier
                
                self._apply_movement_mode(raw_dx, raw_dy, frame_count, best_target)
            else:
                if t_frame - self._last_target_ts > 0.20:
                    smooth_aimer.reset_fatigue()
                    self.ema_target_x = None
                    self.ema_target_y = None
                    self._last_dist = 999.0
                    self.deadzone.reset()

            self._process_jitter(button_held)

            # 5. ОПТИМИЗАЦИЯ: Безопасная отрисовка OpenCV без утечек GDI
            if config.show_debug_window and result.entities:
                if not self._debug_window_initialized:
                    cv2.namedWindow(self._debug_win_name, cv2.WINDOW_AUTOSIZE)
                    x = (config.screen_width - debug_img.shape[1]) // 2
                    y = (config.screen_height - debug_img.shape[0]) // 2
                    cv2.moveWindow(self._debug_win_name, x, y)
                    self._debug_window_initialized = True
                
                cv2.imshow(self._debug_win_name, debug_img)
                cv2.waitKey(1)

            frame_ms = (time.perf_counter() - t_frame) * 1000.0
            self._frame_ms_ema = frame_ms if not self._frame_ms_ema else 0.9 * self._frame_ms_ema + 0.1 * frame_ms
            if frame_ms > 35.0:
                self._slow_frames += 1
                logger.debug("SLOW FRAME %.0f ms", frame_ms)

            frame_count += 1
            elapsed = time.perf_counter() - start_time
            if elapsed > 1.0:
                self.fps = frame_count / elapsed
                fps = self.fps
                start_time = time.perf_counter()
                frame_count = 0

                now_ts = time.perf_counter()
                if now_ts - self._last_summary_ts >= 5.0:
                    st = get_stats()
                    p95 = st.get('p95_ms')
                    logger.info(
                        "STATUS | fps=%.1f frame=%.1fms infer_p95=%s targets=%d held=%s "
                        "lock=%s lock_waits=%d slow=%d dz_active=%s",
                        self.fps, self._frame_ms_ema,
                        f"{p95:.1f}ms" if p95 else "n/a",
                        len(result.entities), button_held,
                        lock_status.value,
                        self._lock_waits, self._slow_frames, self.deadzone.active
                    )
                    self._last_summary_ts = now_ts
                    self._lock_waits = 0
                    self._slow_frames = 0

            time.sleep(0.001)

    def start(self):
        global _aimbot_running, makcu
        if self._running:
            return

        if self.makcu is None:
            if makcu is not None:
                self.makcu = makcu
                self._owns_makcu = False
            else:
                try:
                    self.makcu = Mouse()
                    self._owns_makcu = True
                except Exception as e:
                    logger.error(f"Failed to initialize Mouse: {e}")
                    return
        makcu = self.makcu

        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.smooth_thread = threading.Thread(target=self.smooth_movement_loop, daemon=True)
        self.smooth_thread.start()
        self.aim_thread = threading.Thread(target=self.detection_and_aim_loop, daemon=True)
        self.aim_thread.start()

        if self.jitter_controller is None:
            self.jitter_controller = JitterController(self.makcu) # Убран lock из инициализации

        self._running = True
        _aimbot_running = True
        
        # Вывод конфигурации в лог при старте
        log_startup_config(
            self.target_lock.config,
            self.detector_config,
            ego=self.ego_tracker,
            logger=logger
        )
        
        logger.info(f"Aimbot started in {config.mode} mode.")

    def stop(self):
        global _aimbot_running, makcu
        if not self._running:
            return

        logger.info("Stopping aimbot...")
        self._running = False
        _aimbot_running = False
        self.stop_event.set()

        for t in (self.capture_thread, self.smooth_thread, self.aim_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)

        if self.jitter_controller:
            self.jitter_controller.stop()

        if self._owns_makcu and self.makcu is not None:
            try:
                Mouse.cleanup()
            except Exception:
                pass
            self.makcu = None
            self._owns_makcu = False
            makcu = None

        if config.show_debug_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info("Aimbot stopped")

    def is_running(self):
        return self._running

    def reload_model(self, path=None):
        self.model, self.class_names = load_model(path or config.model_path)
        return self.model

    def get_model_classes(self, path=None):
        _, class_names = load_model(path or config.model_path)
        return [class_names[i] for i in sorted(class_names.keys())]

    def get_model_size(self, path=None):
        try:
            return f"{os.path.getsize(path or config.model_path) / (1024 * 1024):.2f} MB"
        except Exception:
            return "?"


# === Глобальные функции управления ===
_aimbot_instance = None

def start_aimbot():
    global _aimbot_instance
    if _aimbot_instance is None:
        _aimbot_instance = Aimbot()
    _aimbot_instance.start()

def stop_aimbot():
    global _aimbot_instance
    if _aimbot_instance is not None:
        _aimbot_instance.stop()

def is_aimbot_running():
    global _aimbot_instance
    return _aimbot_instance is not None and _aimbot_instance.is_running()

def reload_model(path=None):
    global _aimbot_instance
    if _aimbot_instance is not None:
        return _aimbot_instance.reload_model(path)
    return load_model(path or config.model_path)

def get_model_classes(path=None):
    global _aimbot_instance
    if _aimbot_instance is not None:
        return _aimbot_instance.get_model_classes(path)
    _, class_names = load_model(path or config.model_path)
    return [class_names[i] for i in sorted(class_names.keys())]

def get_model_size(path=None):
    global _aimbot_instance
    if _aimbot_instance is not None:
        return _aimbot_instance.get_model_size(path)
    try:
        return f"{os.path.getsize(path or config.model_path) / (1024 * 1024):.2f} MB"
    except Exception:
        return "?"

def get_fps():
    global _aimbot_instance
    return _aimbot_instance.fps if _aimbot_instance else 0

__all__ = [
    'start_aimbot', 'stop_aimbot', 'is_aimbot_running', 'reload_model',
    'get_model_classes', 'get_model_size', 'get_fps',
    'fps', '_aimbot_running', 'makcu'
]
