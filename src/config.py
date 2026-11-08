import os
import json
import ctypes
from ctypes import wintypes

# Structures
class RECT(ctypes.Structure):
    _fields_ = [
        ("left",   ctypes.c_long),
        ("top",    ctypes.c_long),
        ("right",  ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]

class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize",   ctypes.c_ulong),
        ("rcMonitor", RECT),
        ("rcWork",    RECT),
        ("dwFlags",   ctypes.c_ulong),
    ]

def get_foreground_monitor_resolution():
    # DPI awareness so we get actual pixels
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    user32 = ctypes.windll.user32
    monitor = user32.MonitorFromWindow(user32.GetForegroundWindow(), 2)  # MONITOR_DEFAULTTONEAREST = 2
    mi = MONITORINFO()
    mi.cbSize = ctypes.sizeof(MONITORINFO)

    if ctypes.windll.user32.GetMonitorInfoW(monitor, ctypes.byref(mi)):
        w = mi.rcMonitor.right - mi.rcMonitor.left
        h = mi.rcMonitor.bottom - mi.rcMonitor.top
        return w, h
    else:
        # fallback to primary if anything fails
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

w, h = get_foreground_monitor_resolution()

class Config:
    def __init__(self):
        # --- General Settings ---
        self.region_size = 200
        w, h = get_foreground_monitor_resolution()
        self.screen_width = w
        self.screen_height = h
        self.player_y_offset = 0  # Обновлено на 0, так как теперь используем player_poi_offset
        self.capturer_mode = "NDI"
        self.always_on_aim = False
        self.main_pc_width = 1920
        self.main_pc_height = 1080

        # --- Model and Detection ---
        self.models_dir = "models"
        self.model_path = os.path.join(self.models_dir, "Click here to Load a model")
        self.custom_player_label = "Select a Player Class"  
        self.custom_head_label = "Select a Head Class"  
        self.model_file_size = 0
        self.model_load_error = ""
        self.conf = 0.20  # ОБНОВЛЕНО: Оптимальный баланс для YOLO11 256px
        self.imgsz = 256  # Обновлено под ALO_15k_256.onnx
        self.max_detect = 1
        
        # НОВЫЕ: Настройки точки интереса и трекинга
        self.player_poi_offset = 0.2    # 0.0 = верх бокса, 0.5 = центр, 1.0 = низ
        self.use_tracking = False       # Использовать ли YOLO трекинг (persist=True)

        # --- Mouse / MAKCU ---
        self.selected_mouse_button = 0   # Обновлено на 0 (ЛКМ) как основную кнопку аимбота
        self.makcu_connected = False
        self.makcu_status_msg = "Disconnected"
        self.aim_humanization = 0
        self.in_game_sens = 1.5
        self.button_mask = False

        # --- Trigger Settings ---
        self.trigger_enabled = False
        self.trigger_always_on = False
        self.trigger_button = 1
        self.trigger_radius_px = 12
        self.trigger_delay_ms = 30
        self.trigger_cooldown_ms = 120
        self.trigger_min_conf = 0.35

        # --- Aimbot Mode ---
        self.mode = "normal"
        self.aimbot_running = False
        self.aimbot_status_msg = "Stopped"

        # --- Normal Aim ---
        self.normal_x_speed = 0.8  # ОБНОВЛЕНО: Идеально для формулы tau-гейна
        self.normal_y_speed = 0.8  # ОБНОВЛЕНО: Идеально для формулы tau-гейна
        self.deadzone_enter = 0.8  # НОВЫЙ: Порог входа в мертвую зону

        # --- Target Lock Settings ---
        self.lock_switch_margin = 0.25
        self.lock_switch_frames = 5
        self.lock_lost_frames = 10
        self.lock_match_radius = 0.08
        self.lock_acquire_frames = 2       # кадров подряд для захвата (отсев призраков)
        self.lock_acquire_conf = 0.35      # мин. confidence для ЗАХВАТА/ПЕРЕКЛЮЧЕНИЯ
        self.button_release_frames = 3       # кадров подряд без нажатия для отпускания
        self.short_hold_frames = 6           # удержание короче этого — подозрение на дребезг
        self.aim_latency_s = 0.03        # НОВОЕ: задержка отклика на ваше наведение (сек)
        self.aim_smear_s = 0.04          # НОВОЕ: длительность "размазывания" отклика (сек)

        # --- Bezier Aim ---
        self.bezier_segments = 8
        self.bezier_ctrl_x = 16
        self.bezier_ctrl_y = 16

        # --- Silent Aim ---
        self.silent_segments = 7
        self.silent_ctrl_x = 18
        self.silent_ctrl_y = 18
        self.silent_speed = 3
        self.silent_cooldown = 0.18

        # --- Smooth Aim (WindMouse) ---
        self.smooth_gravity = 9.0
        self.smooth_wind = 3.0
        self.smooth_min_delay = 0.0
        self.smooth_max_delay = 0.002
        self.smooth_max_step = 40.0
        self.smooth_min_step = 2.0
        self.smooth_max_step_ratio = 0.20
        self.smooth_target_area_ratio = 0.06
        self.smooth_reaction_min = 0.05
        self.smooth_reaction_max = 0.21
        self.smooth_close_range = 35
        self.smooth_far_range = 250
        self.smooth_close_speed = 0.8
        self.smooth_far_speed = 1.00
        self.smooth_acceleration = 1.15
        self.smooth_deceleration = 1.05
        self.smooth_fatigue_effect = 1.2
        self.smooth_micro_corrections = 0

        # --- Last error/status for GUI display ---
        self.last_error = ""
        self.last_info = ""

        # --- Debug window toggle ---
        self.show_debug_window = False
        self.log_level = "DEBUG"

        # --- NDI Settings ---
        self.ndi_width = 256
        self.ndi_height = 256
        self.ndi_sources = []
        self.ndi_selected_source = None

        # === Jitter Settings ===
        self.jitter_enabled = True  # Обновлено на True для тестов
        self.jitter_interval = 0.015
        self.hipfire_sens = 0.01
        self.ads_sens = 0.7
        self.fine_tune_offset = -5
        self.jitter_trigger_mode = "both_buttons"

        # === Aim Assist (Console-style) Settings ===
        self.aim_assist_enabled = False
        self.aim_assist_radius = 70
        self.aim_assist_inner_radius = 8
        self.aim_assist_strength = 0.15
        self.aim_assist_slowdown = 0.4
        self.aim_assist_speed_scale = 0.7

        # === WebShaker Jitter Settings ===
        self.jitter_amp_x = 3   # Обновлено на 3 для стабильности
        self.jitter_amp_y = 3   # Обновлено на 3 для стабильности
        self.jitter_speed_ms = 3
        self.jitter_pattern = "square"

    # -- Profile functions --
    def save(self, path="config_profile.json"):
        data = {k.strip(): v for k, v in self.__dict__.items()}
        for k, v in data.items():
            if isinstance(v, str):
                data[k] = v.strip()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path="config_profile.json"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cleaned = {}
            for k, v in data.items():
                key = k.strip()
                if isinstance(v, str):
                    v = v.strip()
                cleaned[key] = v
            self.__dict__.update(cleaned)

    def reset_to_defaults(self):
        self.__init__()

    # --- Utility ---
    def list_models(self):
        if not os.path.exists(self.models_dir):
            return []
        return [f for f in os.listdir(self.models_dir)
                if f.endswith((".engine", ".onnx", ".pt"))]

config = Config()
