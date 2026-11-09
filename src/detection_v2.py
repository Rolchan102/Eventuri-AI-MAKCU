"""Detection & tracking toolkit, v2.12.0.

Модули:
  * DeadzoneFilter  - мёртвая зона с гистерезисом (stateful).
  * DetectorConfig  - валидируемая конфигурация (ошибки -> ValueError, а не молчаливый clamp).
  * build_entities  - ЧИСТАЯ функция: numpy-массивы -> сущности (тестируется без GPU/модели).
  * detect          - инференс (Ultralytics-совместимая модель) + build_entities.
  * TargetLock      - захват цели с удержанием, событиями и мягким удержанием по ID/геометрии.
  * ButtonDebounce / TargetLifecycle - кнопка + замок + логи причин каждого события.
  * draw_debug_overlay - отрисовка, полностью отделена от логики.

Что нового в 2.2 (API полностью совместим с 2.1):
  Скорость
    * detect(): ОДИН перенос с устройства (Boxes.data) вместо 3-4 отдельных;
      есть откат на xyxy/conf/cls/id, если .data недоступна;
    * build_entities(): без индексации numpy-скаляров в цикле (tolist + zip),
      aimed/дистанции векторно, без сортировки для 0-1 сущностей, кэш меток;
    * TargetLock: сравнение по квадратам расстояний (без hypot), ID-поиск через dict;
    * DetectorConfig.infer_classes: необязательный фильтр классов прямо в инференсе
      (меньше работы NMS), по умолчанию выключен.
  Устойчивость
    * "липкий" track_id замка: пропажа ID на кадр (id -> None -> id) больше не REBOUND
      и не ломает матч по ID; REBOUND только при смене метки или смене одного
      известного ID на другой;
    * ID с NaN из трекера безопасно превращаются в None.
  Логи
    * HOLD PRESS/RELEASE и SHORT HOLD - на INFO/WARNING только если за удержание была
      цель (иначе DEBUG; log_idle_holds=True вернёт прежнее поведение);
      short_hold_frames по умолчанию 3.

Что нового в 2.3 (TargetLock, API совместим):
    * экстраполяция позиции цели по скорости (LockConfig.predict) - цель ищется и там,
      где она должна быть, а не только в последней точке; lock.predicted_poi отдаёт
      ожидаемую позицию для наведения во время COASTING;
    * радиус поиска растёт с каждым пропущенным кадром (coast_radius_growth/_max),
      но у сущностей с ЧУЖИМ известным ID остаётся базовым (не воруют цель);
    * COASTING/LOST в логе показывают расстояние до ближайшей видимой сущности:
      "цель была в кадре, но не узнана" отличается от "в кадре никого".

Что нового в 2.4 (API совместим, по умолчанию поведение прежнее):
    * TargetLock.update(..., ego_shift=(dx, dy)) / TargetLifecycle.update(..., ego_shift=...):
      сдвиг неподвижного мира на экране от ВАШЕГО движения мыши с прошлого кадра.
      Замок ищет цель там, куда её сдвинула ваша же камера, а собственная скорость
      цели считается без этого сдвига; при COASTING сдвиги копятся;
    * ego_shift_from_mouse(move_x, move_y, px_per_unit) - перевод вашего move в сдвиг
      (знак минус: мышь вправо -> мир влево); estimate_ego_scale(pairs) - оценка
      px_per_unit методом наименьших квадратов по замерам.
    Пример:  r = life.update(entities, size, held, ego_shift_from_mouse(last_move_x, last_move_y, 1.0))
    где last_move - то, на сколько вы сдвинули мышь на ПРОШЛОМ кадре.

Что нового в 2.5 (против «скачет между целями на дистанции»; по умолчанию поведение прежнее,
кроме size_ratio_max=2.0 и JUMP-диагностики):
    * LockConfig.hard_lock - жёсткий захват: пока цель жива, замок не переключается;
    * LockConfig.size_ratio_max - сосед той же метки, отличающийся по высоте > чем в N раз,
      не считается той же целью (на дистанции цели мелкие, радиус поиска в пикселях
      может быть в несколько раз больше самой цели);
    * DetectorConfig.aim_margin_px - допуск для `aimed` (джиттер прицела не должен менять,
      "под прицелом" цель или нет); поставьте амплитуду джиттера + 1;
    * LockConfig.radius_heights (+ radius_min_frac, radius_max_mult) - БАЛАНС близко/далеко:
      радиус поиска цели пропорционален её высоте (мелкая дальняя - малый радиус, крупная
      ближняя - большой); по умолчанию выкл. (0 = фиксированный match_radius);
    * событие JUMP (+ строка TARGET JUMP в логе): цель найдена далеко от ожидаемой позиции
      (> jump_heights высот) - прямое доказательство подмены цели соседом.

Что нового в 2.6:
    * EgoTracker - готовая сборка ego_shift: ваше наведение + смещение джиттера МЕЖДУ кадрами
      (jitter_offset=jitter.offset_at, latency_s - задержка захвата), в пикселях по px_per_unit:
          ego = EgoTracker(px_per_unit, jitter.offset_at, latency_s=0.03)
          shift = ego.update(aim_move=last_move)      # каждый кадр
          life.update(entities, size, held, shift)

Что нового в 2.7 (без t поведение прежнее, кроме двух защит ниже):
    * TargetLock.update(..., t=время_кадра) / TargetLifecycle.update(..., t=...): скорость цели
      в px/с, экстраполяция по реальному времени (медленные кадры и паузы захвата не ломают
      предсказание); LockConfig.lost_s - потеря цели по времени вместо lost_frames;
    * претендент, перекрывающий текущую цель (голова своего же тела), больше не вызывает
      SWITCHED (LockConfig.skip_overlapping_challengers=True);
    * сущность той же метки с ЧУЖИМ известным ID больше не матчится по "точка внутри бокса";
    * LockConfig.recommended() - конфиг по итогам разборов логов.

Что нового в 2.8 (лог 05.11 14:10-14:12: "близко огонь, на средней перескакивает между целями"):
    * LockConfig.neighbor_guard: радиус поиска цели не больше доли расстояния до ближайшего соседа.
      В логе 10 TARGET JUMP на 1.6-2.1 высоты цели: сосед стоял в ~1.7 высоты, а radius_heights=2
      плюс рост радиуса при COASTING захватывали его за 1-3 пропущенных кадра. В recommended() = 0.5.

Что нового в 2.9 (лог 05.11 14:20-14:22):
    * в строках TARGET COASTING / TARGET LOST - почему ближайшая видимая сущность НЕ принята за цель:
      "не цель: размер x2.40", "за радиусом: 61px > 45px", "защита соседа: ...", "чужой ID 7"
      (LockEvent.reason);
    * LockConfig.relock_after_lost=False - после TARGET LOST замок не хватает другую цель, пока
      кнопку не отпустят (reset()). В recommended() выключено (False); вернуть: recommended(relock_after_lost=True).

Что нового в 2.10 (лог 05.11 15:45-15:56):
    * LockConfig.size_free_heights: бокс, центр которого ближе N * (большая из двух высот) к ожидаемой
      позиции цели, освобождается от проверки размера (size_ratio_max) - это та же цель, обведённая иначе.
      В логе 21 отказ "размер x0.37-0.49 / x2.06-2.45" для бокса в 3-24 px от цели: детектор
      то обводит врага целиком, то половину (перекрытие/присед), высота скачет в ~2-2.5 раза.
      Радиус и защита соседа по-прежнему действуют. В recommended() = 0.5; 0 = выкл.
    Пример:  t = time.perf_counter()   # сразу после захвата кадра
             r = life.update(entities, size, held, ego.update(last_move, t), t)

Подсказка по производительности: если нужна одна цель, use_tracking=False -
TargetLock удерживает цель геометрически и без ID, а model.track() заметно
дороже model().
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

__version__ = "2.12.1"

__all__ = [
    "DeadzoneFilter", "Category", "TrackedEntity", "DetectionResult", "DetectorConfig",
    "build_entities", "detect", "LockConfig", "LockStatus", "LockEventKind", "LockEvent",
    "describe_config", "log_startup_config", "TargetLock", "HoldTransition", "ButtonDebounce", "LifecycleResult", "TargetLifecycle",
    "draw_debug_overlay", "ego_shift_from_mouse", "estimate_ego_scale", "EgoTracker",
]


# ============================================================================
# Deadzone
# ============================================================================
class DeadzoneFilter:
    """Мёртвая зона с гистерезисом: вход при |v| >= enter, выход при |v| < exit.

    exit_ratio в (0, 1]; при 1.0 гистерезис отключён.
    rescale=True вычитает порог (плавный старт без скачка на границе).
    Фильтр хранит состояние: один экземпляр на один источник сигнала.
    """

    __slots__ = ("enter", "exit", "radial", "snap_axes", "rescale", "_active")

    def __init__(self, enter: float, exit_ratio: float = 0.6, *,
                 radial: bool = True, snap_axes: bool = False,
                 rescale: bool = False) -> None:
        if not (math.isfinite(enter) and enter > 0):
            raise ValueError(f"enter must be finite and > 0, got {enter!r}")
        if not (math.isfinite(exit_ratio) and 0 < exit_ratio <= 1):
            raise ValueError(f"exit_ratio must be in (0, 1], got {exit_ratio!r}")
        self.enter = float(enter)
        self.exit = self.enter * exit_ratio
        self.radial = radial
        self.snap_axes = snap_axes
        self.rescale = rescale
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False

    def apply(self, dx: float, dy: float) -> tuple[float, float]:
        if not (math.isfinite(dx) and math.isfinite(dy)):
            self._active = False
            return 0.0, 0.0

        mag = math.hypot(dx, dy) if self.radial else max(abs(dx), abs(dy))
        self._active = mag >= (self.exit if self._active else self.enter)
        if not self._active:
            return 0.0, 0.0

        if self.snap_axes:
            if abs(dx) < self.exit:
                dx = 0.0
            if abs(dy) < self.exit:
                dy = 0.0

        if self.rescale:
            dx = math.copysign(max(abs(dx) - self.exit, 0.0), dx)
            dy = math.copysign(max(abs(dy) - self.exit, 0.0), dy)
        return dx, dy


# ============================================================================
# Data structures
# ============================================================================
class Category(str, Enum):
    PRIORITY = "priority"
    STANDARD = "standard"


@dataclass(frozen=True, slots=True)
class TrackedEntity:
    track_id: Optional[int]
    label: str
    category: Category
    confidence: float
    bbox: tuple[float, float, float, float]    # x1, y1, x2, y2, исходные float
    poi: tuple[float, float]                   # опорная точка (x, y)
    height: float
    distance_px: float
    distance_norm: float                       # distance_px / диагональ кадра
    aimed: bool = False                        # опорная точка прицела внутри бокса


@dataclass(frozen=True, slots=True)
class DetectionResult:
    entities: list[TrackedEntity]              # отсортированы: ближе к центру, затем увереннее
    observed_labels: frozenset[str]
    tracking_lost: bool = False                # трекинг включён, но трекер не выдал ID


_EMPTY_RESULT = DetectionResult([], frozenset(), False)


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    conf_threshold: float = 0.35
    poi_offset: float = 0.5                    # 0 = верх бокса, 0.5 = центр, 1 = низ
    use_tracking: bool = False
    priority_labels: frozenset[str] = frozenset()
    relevant_labels: Optional[frozenset[str]] = None   # None = все метки релевантны
    ref_center: Optional[tuple[float, float]] = None   # None = центр кадра
    infer_classes: Optional[tuple[int, ...]] = None    # фильтр классов в инференсе (меньше NMS);
    #                                                    ВНИМАНИЕ: остальные классы не попадут и в observed_labels
    aim_margin_px: float = 0.0                 # допуск для aimed: бокс расширяется на N px (поставьте
    #                                            амплитуду джиттера + 1, чтобы aimed не мигал)

    def __post_init__(self) -> None:
        for name in ("conf_threshold", "poi_offset"):
            v = getattr(self, name)
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v!r}")
        if self.ref_center is not None and not all(math.isfinite(c) for c in self.ref_center):
            raise ValueError(f"ref_center must be finite, got {self.ref_center!r}")
        if self.infer_classes is not None and not all(isinstance(c, int) for c in self.infer_classes):
            raise ValueError("infer_classes must be a tuple of int class ids")
        if not (math.isfinite(self.aim_margin_px) and self.aim_margin_px >= 0):
            raise ValueError(f"aim_margin_px must be >= 0, got {self.aim_margin_px!r}")
        # приоритетные метки всегда релевантны
        if self.relevant_labels is not None and not self.priority_labels <= self.relevant_labels:
            object.__setattr__(self, "relevant_labels",
                               self.relevant_labels | self.priority_labels)


# ============================================================================
# Pure core
# ============================================================================
def build_entities(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    classes: np.ndarray,
    ids: Optional[np.ndarray],
    frame_size: tuple[int, int],               # (height, width)
    label_map: Mapping[int, str],
    config: DetectorConfig,
) -> DetectionResult:
    """Чистая функция без обращения к модели и GPU."""
    h, w = frame_size
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid frame size {frame_size!r}")

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    confs = np.asarray(confs, dtype=np.float64).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    n = len(boxes_xyxy)
    if not (len(confs) == len(classes) == n):
        raise ValueError("boxes/confs/classes/ids length mismatch")
    if ids is not None:
        ids = np.asarray(ids, dtype=np.float64).reshape(-1)
        if len(ids) != n:
            raise ValueError("boxes/confs/classes/ids length mismatch")
    if n == 0:
        return _EMPTY_RESULT

    if n <= _SMALL_N:
        return _build_small(boxes_xyxy, confs, classes, ids, h, w, label_map, config)
    return _build_vec(boxes_xyxy, confs, classes, ids, h, w, label_map, config)


# Порог выбора пути (замер на CPU: до ~12 сущностей чистый Python быстрее векторного
# numpy - ~20 numpy-вызовов на крошечных массивах стоят дороже цикла).
_SMALL_N = 12


def _finish(entities: list[TrackedEntity], observed: set[str], use_tracking: bool,
            has_ids: bool) -> DetectionResult:
    if len(entities) > 1:
        entities.sort(key=lambda e: (e.distance_px, -e.confidence))
    return DetectionResult(entities, frozenset(observed),
                           use_tracking and not has_ids and bool(entities))


def _build_small(boxes: np.ndarray, confs: np.ndarray, classes: np.ndarray,
                 ids: Optional[np.ndarray], h: int, w: int,
                 label_map: Mapping[int, str], config: DetectorConfig) -> DetectionResult:
    """Путь для малого числа сущностей: один tolist() на массив, дальше чистый Python."""
    rows = boxes.tolist()
    conf_l = confs.tolist()
    cls_l = classes.astype(np.int64).tolist()
    ids_l = ids.tolist() if ids is not None else None

    cx, cy = config.ref_center if config.ref_center is not None else (w / 2.0, h / 2.0)
    diag = math.hypot(w, h)
    off = config.poi_offset
    am = config.aim_margin_px
    relevant, priority = config.relevant_labels, config.priority_labels
    PRIO, STD = Category.PRIORITY, Category.STANDARD
    isfinite, hypot = math.isfinite, math.hypot
    names: dict[int, str] = {}

    entities: list[TrackedEntity] = []
    observed: set[str] = set()
    for i in range(len(rows)):
        x1, y1, x2, y2 = rows[i]
        conf = conf_l[i]
        if not (x2 > x1 and y2 > y1 and isfinite(x1) and isfinite(y1)
                and isfinite(x2) and isfinite(y2) and isfinite(conf)):
            continue
        cid = cls_l[i]
        label = names.get(cid)
        if label is None:
            label = names[cid] = label_map.get(cid, f"class_{cid}")
        observed.add(label)
        if relevant is not None and label not in relevant:
            continue
        tid = None
        if ids_l is not None:
            t = ids_l[i]
            if isfinite(t) and t >= 0:
                tid = int(t)
        ht = y2 - y1
        px = (x1 + x2) * 0.5
        py = y1 + ht * off
        d = hypot(px - cx, py - cy)
        entities.append(TrackedEntity(
            tid, label, PRIO if label in priority else STD, conf,
            (x1, y1, x2, y2), (px, py), ht, d, d / diag,
            x1 - am <= cx <= x2 + am and y1 - am <= cy <= y2 + am))
    return _finish(entities, observed, config.use_tracking, ids is not None)


def _build_vec(boxes: np.ndarray, confs: np.ndarray, classes: np.ndarray,
               ids: Optional[np.ndarray], h: int, w: int,
               label_map: Mapping[int, str], config: DetectorConfig) -> DetectionResult:
    """Векторный путь для большого числа сущностей."""
    x1, y1, x2, y2 = boxes.T
    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(confs) & (x2 > x1) & (y2 > y1)
    if not valid.all():
        boxes, confs, classes = boxes[valid], confs[valid], classes[valid]
        if ids is not None:
            ids = ids[valid]
        if len(boxes) == 0:
            return _EMPTY_RESULT
        x1, y1, x2, y2 = boxes.T

    cx, cy = config.ref_center if config.ref_center is not None else (w / 2.0, h / 2.0)
    diag = math.hypot(w, h)

    heights = y2 - y1
    poi_x = (x1 + x2) * 0.5
    poi_y = y1 + heights * config.poi_offset
    dist = np.hypot(poi_x - cx, poi_y - cy)
    am = config.aim_margin_px
    aimed = (x1 - am <= cx) & (cx <= x2 + am) & (y1 - am <= cy) & (cy <= y2 + am)

    cls_l = classes.astype(np.int64).tolist()
    conf_l, d_l, h_l = confs.tolist(), dist.tolist(), heights.tolist()
    x1_l, y1_l, x2_l, y2_l = x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()
    px_l, py_l, aim_l = poi_x.tolist(), poi_y.tolist(), aimed.tolist()
    ids_l = (np.nan_to_num(ids, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.int64).tolist()
             if ids is not None else None)

    relevant, priority = config.relevant_labels, config.priority_labels
    PRIO, STD = Category.PRIORITY, Category.STANDARD
    names: dict[int, str] = {}

    entities: list[TrackedEntity] = []
    observed: set[str] = set()
    for i in range(len(cls_l)):
        cid = cls_l[i]
        label = names.get(cid)
        if label is None:
            label = names[cid] = label_map.get(cid, f"class_{cid}")
        observed.add(label)
        if relevant is not None and label not in relevant:
            continue
        tid = ids_l[i] if ids_l is not None else -1
        entities.append(TrackedEntity(
            tid if tid >= 0 else None, label, PRIO if label in priority else STD,
            conf_l[i], (x1_l[i], y1_l[i], x2_l[i], y2_l[i]), (px_l[i], py_l[i]),
            h_l[i], d_l[i], d_l[i] / diag, aim_l[i]))
    return _finish(entities, observed, config.use_tracking, ids is not None)


# ============================================================================
# Inference
# ============================================================================
def _to_numpy(t: Any) -> np.ndarray:
    """Tensor (torch) или уже ndarray -> ndarray."""
    if hasattr(t, "cpu"):
        t = t.cpu()
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def _extract(b: Any, use_tracking: bool):
    """(xyxy, conf, cls, ids|None) из Ultralytics Boxes.

    Быстрый путь: Boxes.data = [x1,y1,x2,y2,(id),conf,cls] - ОДИН перенос с устройства.
    Если .data недоступна или форма неожиданная - читаем поля по отдельности.
    """
    data = getattr(b, "data", None)
    if data is not None:
        arr = _to_numpy(data)
        if arr.ndim == 2 and arr.shape[1] in (6, 7):
            ids = arr[:, 4] if (use_tracking and arr.shape[1] == 7) else None
            return arr[:, :4], arr[:, -2], arr[:, -1], ids
    ids = _to_numpy(b.id) if (use_tracking and getattr(b, "id", None) is not None) else None
    return _to_numpy(b.xyxy), _to_numpy(b.conf), _to_numpy(b.cls), ids


def detect(model: Any, frame: np.ndarray, label_map: Mapping[int, str],
           config: DetectorConfig = DetectorConfig()) -> DetectionResult:
    """Инференс + сборка сущностей.

    Трекинг (persist=True) хранит состояние внутри модели: используйте отдельный
    экземпляр модели на каждый видеопоток. model.track() заметно дороже model();
    для одной цели TargetLock справляется и без него (use_tracking=False).
    """
    if frame is None or frame.ndim < 2:
        raise ValueError("frame must be an image array")

    kw: dict[str, Any] = {"conf": config.conf_threshold, "verbose": False}
    if config.infer_classes is not None:
        kw["classes"] = list(config.infer_classes)
    if config.use_tracking:
        results = model.track(frame, persist=True, **kw)
    else:
        results = model(frame, **kw)

    if not results:
        return _EMPTY_RESULT
    b = results[0].boxes
    if b is None or len(b) == 0:
        return _EMPTY_RESULT

    xyxy, conf, cls, ids = _extract(b, config.use_tracking)
    return build_entities(xyxy, conf, cls, ids, frame.shape[:2], label_map, config)


# ============================================================================
# Target lock (стабильный выбор цели)
# ============================================================================
@dataclass(frozen=True, slots=True)
class LockConfig:
    switch_margin: float = 0.25    # новая цель должна быть на 25% ближе к прицелу
    switch_frames: int = 5         # ...и удерживать преимущество N кадров подряд
    lost_frames: int = 10          # сколько кадров "дрейфовать" без цели до сброса
    match_radius: float = 0.08     # радиус поиска той же цели, доля диагонали кадра
    acquire_frames: int = 2        # кадров подряд для захвата (отсев одиночных ложных)
    acquire_conf: float = 0.35     # мин. confidence для ЗАХВАТА/ПЕРЕКЛЮЧЕНИЯ; удержание - любой
    coast_radius_growth: float = 0.5   # радиус растёт на 50% match_radius за каждый пропущенный кадр...
    coast_radius_max: float = 3.0      # ...но не более чем в 3 раза
    predict: bool = True           # экстраполировать позицию цели по скорости (к ней же ищем)
    hard_lock: bool = False        # жёсткий захват: НИКАКИХ переключений на другую цель, пока эта
    #                                жива (цель теряется только по lost_frames или reset())
    size_ratio_max: float = 2.0    # той же метки: бокс-кандидат по высоте не больше чем в N раз
    #                                отличается от цели (отсекает соседей другого размера); 0 = выкл.
    radius_heights: float = 0.0    # БАЛАНС БЛИЗКО/ДАЛЕКО: радиус поиска = N высот цели вместо фиксированного
    #                                match_radius (0 = выкл.). Мелкая дальняя цель -> малый радиус (соседи не
    #                                подменяют), крупная ближняя -> большой (быстро уходит в пикселях)
    radius_min_frac: float = 0.04  # ...но не меньше этой доли диагонали кадра
    radius_max_mult: float = 2.0   # ...и не больше match_radius * N
    jump_heights: float = 1.5      # событие JUMP, если найденная цель смещена от ожидаемой позиции
    #                                больше чем на N высот цели (подозрение на соседа); 0 = выкл.
    lost_s: float = 0.0            # > 0 и в update() передаётся t: цель теряется, если не видна дольше
    #                                lost_s секунд (вместо lost_frames; кадры разной длины не мешают)
    skip_overlapping_challengers: bool = True   # претендент, перекрывающий цель (голова своего же тела),
    #                                             не может её "переманить" (ложный SWITCHED)
    relock_after_lost: bool = True # False: после LOST (цель не вернулась за lost_s/lost_frames) НЕ захватывать
    #                                другую цель до reset() (отпускания кнопки) - без "перескока" на соседа
    #                                после потери; цена - после смерти цели нужно нажать кнопку заново
    size_free_heights: float = 0.0  # > 0: кандидат ближе N * max(высота цели, высота кандидата) к ожидаемой
    #                                позиции - без проверки size_ratio_max (детектор обвёл ту же цель
    #                                наполовину/целиком). Радиус и neighbor_guard действуют. 0 = выкл.
    neighbor_guard: float = 0.0    # > 0: радиус поиска цели (и при COASTING тоже) не больше N * расстояния
    #                                до ближайшего СОСЕДА в последнем кадре, где цель была видна. Против
    #                                "перескакивает между целями" на средней дистанции: враги стоят в 1.5-2
    #                                высотах друг от друга, а radius_heights/рост радиуса дотягиваются до соседа.
    #                                0.5 = граница посередине между целью и соседом; 0 = выкл.

    def __post_init__(self) -> None:
        if not (math.isfinite(self.switch_margin) and 0.0 <= self.switch_margin < 1.0):
            raise ValueError("switch_margin must be in [0, 1)")
        if self.switch_frames < 1 or self.acquire_frames < 1 or self.lost_frames < 0:
            raise ValueError("switch_frames/acquire_frames >= 1 and lost_frames >= 0 required")
        if not (math.isfinite(self.match_radius) and self.match_radius > 0):
            raise ValueError("match_radius must be > 0")
        if not (math.isfinite(self.acquire_conf) and 0.0 <= self.acquire_conf <= 1.0):
            raise ValueError("acquire_conf must be in [0, 1]")
        if not (math.isfinite(self.coast_radius_growth) and self.coast_radius_growth >= 0):
            raise ValueError("coast_radius_growth must be >= 0")
        if not (math.isfinite(self.coast_radius_max) and self.coast_radius_max >= 1.0):
            raise ValueError("coast_radius_max must be >= 1")
        if not (math.isfinite(self.size_ratio_max) and (self.size_ratio_max == 0 or self.size_ratio_max >= 1.0)):
            raise ValueError("size_ratio_max must be 0 (off) or >= 1")
        if not (math.isfinite(self.jump_heights) and self.jump_heights >= 0):
            raise ValueError("jump_heights must be >= 0")
        if not (math.isfinite(self.radius_heights) and self.radius_heights >= 0):
            raise ValueError("radius_heights must be >= 0")
        if not (math.isfinite(self.radius_max_mult) and self.radius_max_mult >= 1.0):
            raise ValueError("radius_max_mult must be >= 1")
        if not (math.isfinite(self.radius_min_frac) and 0 < self.radius_min_frac
                <= self.match_radius * self.radius_max_mult):
            raise ValueError("radius_min_frac must be in (0, match_radius * radius_max_mult]")
        if not (math.isfinite(self.lost_s) and self.lost_s >= 0):
            raise ValueError("lost_s must be >= 0")
        if not (math.isfinite(self.size_free_heights) and self.size_free_heights >= 0):
            raise ValueError("size_free_heights must be >= 0")
        if not (math.isfinite(self.neighbor_guard) and 0 <= self.neighbor_guard <= 1):
            raise ValueError("neighbor_guard must be in [0, 1]")

    @classmethod
    def recommended(cls, **overrides: Any) -> "LockConfig":
        """Конфиг по итогам разборов логов 04-05.11.2026 (tracking OFF, одна цель)."""
        base: dict[str, Any] = dict(match_radius=0.10, radius_heights=2.0, hard_lock=True,
                                    size_ratio_max=2.0, lost_frames=15, lost_s=0.15,
                                    neighbor_guard=0.5, relock_after_lost=False,
                                    size_free_heights=0.5)
        base.update(overrides)
        return cls(**base)


class LockStatus(str, Enum):
    IDLE = "idle"          # цели нет
    PENDING = "pending"    # кандидат набирает acquire_frames
    TRACKING = "tracking"  # цель видна в этом кадре
    COASTING = "coasting"  # цель пропала, держим последнюю позицию


class LockEventKind(str, Enum):
    ACQUIRED = "acquired"          # новый захват
    SWITCHED = "switched"          # РЕАЛЬНОЕ переключение на другую цель (претендент победил)
    REBOUND = "rebound"            # та же цель, но сменилась метка или ID (Тело<->Голова, новый трек)
    COAST_START = "coast_start"    # цель пропала из детекций, держим последнюю позицию
    REACQUIRED = "reacquired"      # цель вернулась после COASTING
    LOST_TIMEOUT = "lost_timeout"  # цель не вернулась за lost_frames: захват сброшен
    RESET = "reset"                # внешний reset() (например, отпустили кнопку)
    JUMP = "jump"                  # найденная цель далеко от ожидаемой позиции (возможна подмена соседом);
    #                                nearest_px = смещение в px


@dataclass(frozen=True, slots=True)
class LockEvent:
    """Решение замка за один кадр. missed_frames - значение счётчика ДО его
    обнуления/сброса (в отличие от lock.missed_frames после reset).
    nearest_px (COAST_START/LOST_TIMEOUT) - расстояние от ожидаемой позиции цели до
    ближайшей ВИДИМОЙ сущности: маленькое = цель, вероятно, в кадре, но не узнана;
    None = в кадре никого не было."""
    kind: LockEventKind
    prev: Optional[TrackedEntity] = None
    new: Optional[TrackedEntity] = None
    missed_frames: int = 0
    nearest_px: Optional[float] = None
    reason: Optional[str] = None   # COAST_START/LOST_TIMEOUT: почему ближайшая видимая сущность НЕ принята
    #                                за цель ("размер x2.4", "за радиусом 61px > 45px", "защита соседа ...")


class TargetLock:
    """Захват цели с удержанием. Один экземпляр на один видеопоток.

    Показывайте "TARGET LOST" только при status == IDLE; COASTING - это
    кратковременная потеря детекции, цель всё ещё удерживается.

    Захват: цель под прицелом (aimed) важнее, затем ближайшая; нужен confidence
    >= acquire_conf и acquire_frames кадров подряд. Удержание: по "липкому"
    track_id (последний известный ID цели переживает кадры, где трекер вернул None),
    а при его смене/потере - геометрически: сущность в радиусе match_radius от
    последней ИЛИ ожидаемой позиции цели, либо одна из точек внутри бокса другой;
    метка НЕ обязана совпадать (Тело <-> Голова).
    Ожидаемая позиция - экстраполяция по скорости цели (predict=True). Пока цель
    пропала, радиус поиска растёт с каждым кадром (coast_radius_growth, до
    coast_radius_max), но только для сущностей без ИДЕНТИЧНОСТИ-конфликта: у чужого
    известного ID радиус остаётся базовым (трекер сказал "это другой объект").
    Удерживается цель ЛЮБОГО confidence (гистерезис: захват строгий, удержание
    мягкое), поэтому "та же цель" может перетечь на сущность с conf < acquire_conf -
    это REBOUND, а не SWITCHED. REBOUND - только смена метки либо смена одного
    ИЗВЕСТНОГО ID на другой (id -> None -> id событием не является).
    Переключение (SWITCHED): другая цель устойчиво (switch_frames) лучше текущей
    и проходит фильтр acquire_conf.
    Пока цель пропала (до lost_frames), update() возвращает ПОСЛЕДНЮЮ известную
    (а ожидаемую позицию для наведения даёт lock.predicted_poi).

    Диагностика: после update()/reset() lock.events - решения замка за этот
    вызов (ACQUIRED/SWITCHED/REBOUND/COAST_START/REACQUIRED/LOST_TIMEOUT/RESET).
    """

    def __init__(self, config: LockConfig = LockConfig()) -> None:
        self.config = config
        self._events: list[LockEvent] = []
        self._max_shift = math.inf
        self._timed = False            # скорость в px/с (update получает t) или в px/кадр
        self._t_now: Optional[float] = None
        self._clear()

    def _clear(self) -> None:
        self._target: Optional[TrackedEntity] = None
        self._t_obs: Optional[float] = None         # время последнего наблюдения цели (режим t)
        self._sep = math.inf                        # расстояние от цели до ближайшего соседа, px
        self._blocked = False                       # relock_after_lost=False: после LOST ждём reset()
        self._known_id: Optional[int] = None       # последний известный track_id цели
        self._vel: Optional[tuple[float, float]] = None   # скорость опорной точки, px/кадр
        self._ego_acc = (0.0, 0.0)                 # сдвиг мира от нашей камеры с последнего наблюдения
        self._missed = 0
        self._challenger_streak = 0
        self._pending: Optional[TrackedEntity] = None
        self._pending_streak = 0

    def reset(self) -> None:
        """Внешний сброс. Если цель была захвачена - событие RESET с реальным missed_frames."""
        self._events = []
        if self._target is not None:
            self._events.append(LockEvent(LockEventKind.RESET, prev=self._target,
                                          missed_frames=self._missed))
        self._clear()   # заодно снимает блокировку перезахвата

    @property
    def events(self) -> tuple[LockEvent, ...]:
        return tuple(self._events)

    @property
    def locked(self) -> bool:
        return self._target is not None

    @property
    def coasting(self) -> bool:
        return self._target is not None and self._missed > 0

    @property
    def missed_frames(self) -> int:
        return self._missed

    @property
    def timed(self) -> bool:
        """True, если последний update() получил t (скорость в px/с, потеря по lost_s)."""
        return self._timed

    @property
    def status(self) -> LockStatus:
        if self._target is not None:
            return LockStatus.COASTING if self._missed > 0 else LockStatus.TRACKING
        return LockStatus.PENDING if self._pending is not None else LockStatus.IDLE

    @property
    def predicted_poi(self) -> Optional[tuple[float, float]]:
        """Ожидаемая позиция цели в ТЕКУЩЕМ кадре (во время COASTING - с экстраполяцией)."""
        t = self._target
        if t is None:
            return None
        ax, ay = self._anchor(t, self._elapsed(self._t_now, self._missed))
        return ax + self._ego_acc[0], ay + self._ego_acc[1]

    def _elapsed(self, now: Optional[float], frames: int) -> float:
        """Сколько прошло с последнего наблюдения цели: секунды (режим t) или кадры."""
        if self._timed and now is not None and self._t_obs is not None:
            return max(now - self._t_obs, 0.0)
        return float(frames)

    @staticmethod
    def _acquire_key(e: TrackedEntity):
        return (not e.aimed, e.distance_px, -e.confidence)

    def _radius(self, e: TrackedEntity, diag: float) -> float:
        """Базовый радиус поиска той же цели, px. radius_heights > 0: пропорционален высоте
        цели в пределах [radius_min_frac * diag, radius_max_mult * match_radius * diag]."""
        cfg = self.config
        base = cfg.match_radius * diag
        if cfg.radius_heights <= 0:
            return base
        return min(max(cfg.radius_heights * e.height, cfg.radius_min_frac * diag),
                   cfg.radius_max_mult * base)

    def _emit(self, kind: LockEventKind, prev=None, new=None, missed: int = 0,
              nearest: Optional[float] = None, reason: Optional[str] = None) -> None:
        self._events.append(LockEvent(kind, prev, new, missed, nearest, reason))

    def _why_not(self, prev: TrackedEntity, entities: Sequence[TrackedEntity],
                 anchor: tuple[float, float], shift: tuple[float, float],
                 limit: float, grown: float, free_limit: float, free_grown: float,
                 cap: Optional[float]) -> tuple[Optional[float], Optional[str]]:
        """(расстояние до ближайшей видимой сущности от ожидаемой позиции, почему она НЕ цель).
        Только для диагностики: повторяет проверки _match для одной ближайшей сущности."""
        if not entities:
            return None, None
        ax, ay = anchor
        e = min(entities, key=lambda x: (x.poi[0] - ax) ** 2 + (x.poi[1] - ay) ** 2)
        near = math.hypot(e.poi[0] - ax, e.poi[1] - ay)
        cfg = self.config
        sr = cfg.size_ratio_max
        lx0, ly0 = prev.poi[0] + shift[0], prev.poi[1] + shift[1]
        d_size = min(near, math.hypot(e.poi[0] - lx0, e.poi[1] - ly0))
        free = cfg.size_free_heights * max(prev.height, e.height)
        if sr > 0 and e.label == prev.label and prev.height > 0                 and (cfg.size_free_heights <= 0 or d_size > free):
            r = e.height / prev.height
            if r > sr or r < 1.0 / sr:
                return near, f"размер x{r:.2f} (предел {sr:g})"
        lx, ly = prev.poi[0] + shift[0], prev.poi[1] + shift[1]
        d = min(near, math.hypot(e.poi[0] - lx, e.poi[1] - ly))
        conflict = self._known_id is not None and e.track_id is not None
        if conflict:
            if e.label == prev.label:
                return near, f"чужой ID {e.track_id}"
            r_now, r_free = limit, free_limit
        else:
            r_now, r_free = grown, free_grown
        if cap is not None and d > r_now and d <= r_free:
            return near, (f"защита соседа: {d:.0f}px > {r_now:.0f}px "
                          f"(сосед был в {self._sep:.0f}px)")
        if d > r_now:
            return near, f"за радиусом: {d:.0f}px > {r_now:.0f}px"
        return near, "в радиусе, но не принята"

    def _anchor(self, prev: TrackedEntity, elapsed: float) -> tuple[float, float]:
        """Ожидаемая позиция цели через `elapsed` (кадров или секунд, см. _elapsed)
        после последнего наблюдения."""
        v = self._vel
        if not self.config.predict or v is None or elapsed <= 0:
            return prev.poi
        sx, sy = v[0] * elapsed, v[1] * elapsed
        n = math.hypot(sx, sy)
        if n > self._max_shift:                      # защита от разгона экстраполяции
            sx, sy = sx * self._max_shift / n, sy * self._max_shift / n
        return prev.poi[0] + sx, prev.poi[1] + sy

    @staticmethod
    def _nearest(entities: Sequence[TrackedEntity], pos: tuple[float, float]) -> Optional[float]:
        if not entities:
            return None
        return min(math.hypot(e.poi[0] - pos[0], e.poi[1] - pos[1]) for e in entities)

    @staticmethod
    def _match(prev: TrackedEntity, known_id: Optional[int], entities: Sequence[TrackedEntity],
               limit_sq: float, grown_sq: float, anchor: tuple[float, float],
               shift: tuple[float, float] = (0.0, 0.0),
               size_ratio: float = 0.0,
               size_free: float = 0.0) -> Optional[TrackedEntity]:
        """Та же цель в новом кадре: сначала по ID, затем геометрически.

        shift - смещение НЕПОДВИЖНОГО мира на экране от нашего собственного движения
        камеры с момента последнего наблюдения: последняя позиция цели сдвигается на
        него (anchor уже включает и shift, и собственную скорость цели)."""
        if known_id is not None:
            for e in entities:
                if e.track_id == known_id:
                    return e
        sx, sy = shift
        lx, ly = prev.poi[0] + sx, prev.poi[1] + sy
        ax, ay = anchor
        moved = ax != lx or ay != ly
        pb = prev.bbox
        pb0, pb1, pb2, pb3 = pb[0] + sx, pb[1] + sy, pb[2] + sx, pb[3] + sy
        ph = prev.height
        inv = 1.0 / size_ratio if size_ratio > 0 else 0.0
        best, best_key = None, None
        for e in entities:
            ex, ey = e.poi
            d2 = (ex - lx) ** 2 + (ey - ly) ** 2
            if moved:
                d2 = min(d2, (ex - ax) ** 2 + (ey - ay) ** 2)
            if size_ratio > 0 and e.label == prev.label and ph > 0:
                f = size_free * max(ph, e.height)  # почти совпадает с целью - тот же враг, обведённый иначе
                if size_free <= 0 or d2 > f * f:
                    r = e.height / ph             # сосед другого размера - не та же цель
                    if r > size_ratio or r < inv:
                        continue
            # чужой известный ID (known_id есть, ID совпадающий уже отброшен выше) -> базовый радиус
            # и без поблажки "точка внутри бокса", если метка та же: трекер прямо сказал, что это
            # другой объект. Другая метка с другим ID - норма (Ultralytics ведёт треки по классам,
            # при смене Тело<->Голова ID меняется), ей поблажка остаётся.
            conflict = known_id is not None and e.track_id is not None
            if d2 > (limit_sq if conflict else grown_sq):
                if conflict and e.label == prev.label:
                    continue
                eb = e.bbox
                if not ((pb0 <= ex <= pb2 and pb1 <= ey <= pb3)
                        or (eb[0] <= lx <= eb[2] and eb[1] <= ly <= eb[3])):
                    continue
            key = (e.label != prev.label, d2)
            if best_key is None or key < best_key:
                best, best_key = e, key
        return best

    def _acquire(self, cands: Sequence[TrackedEntity],
                 diag: float) -> Optional[TrackedEntity]:
        if not cands:
            self._pending, self._pending_streak = None, 0
            return None
        pend = self._pending
        lim = self._radius(pend, diag) if pend is not None else 0.0
        cont = (self._match(pend, pend.track_id, cands, lim * lim, lim * lim, pend.poi,
                            (0.0, 0.0), self.config.size_ratio_max, self.config.size_free_heights)
                if pend is not None else None)
        if cont is not None:
            self._pending, self._pending_streak = cont, self._pending_streak + 1
        else:
            self._pending, self._pending_streak = min(cands, key=self._acquire_key), 1
        if self._pending_streak >= self.config.acquire_frames:
            self._target, self._pending, self._pending_streak = self._pending, None, 0
            self._known_id = self._target.track_id
            self._vel = None
            self._t_obs = self._t_now
            self._missed = self._challenger_streak = 0
            self._emit(LockEventKind.ACQUIRED, new=self._target)
            return self._target
        return None

    def update(self, entities: Sequence[TrackedEntity],
               frame_size: tuple[int, int],
               ego_shift: tuple[float, float] = (0.0, 0.0),
               t: Optional[float] = None) -> Optional[TrackedEntity]:
        """Возвращает текущую (или последнюю известную) цель либо None.
        frame_size = (height, width). События кадра - в self.events.

        ego_shift = (dx, dy) - на сколько пикселей экрана сдвинулся НЕПОДВИЖНЫЙ мир
        между прошлым и этим кадром из-за вашего собственного движения мыши/камеры
        (см. ego_shift_from_mouse). По умолчанию (0, 0) - поведение как раньше.
        Во время COASTING сдвиги накапливаются.

        t - время съёмки кадра в секундах (например time.perf_counter() при захвате).
        С ним скорость цели считается в px/с, а экстраполяция - по реальному времени:
        медленный кадр (100 мс) или пауза захвата не ломают предсказание позиции.
        Потеря цели по времени - LockConfig.lost_s. Без t - прежний режим "по кадрам"."""
        self._events = []
        cfg = self.config
        if self._blocked:                            # relock_after_lost=False: до отпускания кнопки - ничего
            return None
        if t is not None and not math.isfinite(t):
            t = None
        timed = t is not None
        if timed != self._timed:                     # сменился режим - единицы скорости другие
            self._timed, self._vel = timed, None
        self._t_now = t
        diag = math.hypot(frame_size[0], frame_size[1])
        self._max_shift = 0.5 * diag
        min_conf = cfg.acquire_conf
        cands = [e for e in entities if e.confidence >= min_conf]

        if self._target is None:
            return self._acquired(self._acquire(cands, diag), entities)

        ex_, ey_ = ego_shift
        if not (math.isfinite(ex_) and math.isfinite(ey_)):
            ex_ = ey_ = 0.0
        acc = (self._ego_acc[0] + ex_, self._ego_acc[1] + ey_)   # сдвиг мира с последнего наблюдения

        prev = self._target
        limit = self._radius(prev, diag)
        k = self._missed                               # кадров подряд без цели ДО этого
        mult = min(1.0 + cfg.coast_radius_growth * k, cfg.coast_radius_max)
        grown = limit * mult
        free_limit, free_grown, cap = limit, grown, None
        if cfg.neighbor_guard > 0 and self._sep < math.inf:
            # не дотягиваемся до соседа: граница - доля расстояния до него (но не меньше 1/4 высоты)
            cap = max(cfg.neighbor_guard * self._sep, 0.25 * prev.height)
            limit, grown = min(limit, cap), min(grown, cap)
        limit_sq = limit * limit
        elapsed = self._elapsed(t, k + 1)              # кадров или секунд с последнего наблюдения
        ax, ay = self._anchor(prev, elapsed)           # позиция по собственной скорости цели
        anchor = (ax + acc[0], ay + acc[1])            # + сдвиг от нашего движения
        current = self._match(prev, self._known_id, entities, limit_sq, grown * grown, anchor, acc,
                              cfg.size_ratio_max, cfg.size_free_heights)

        if current is None:
            self._missed += 1
            self._ego_acc = acc
            self._challenger_streak = 0
            if timed and cfg.lost_s > 0 and self._t_obs is not None:
                lost = elapsed > cfg.lost_s
            else:
                lost = self._missed > cfg.lost_frames
            if lost or self._missed == 1:
                near, why = self._why_not(prev, entities, anchor, acc, limit, grown,
                                          free_limit, free_grown, cap)
            if lost:
                self._emit(LockEventKind.LOST_TIMEOUT, prev=prev, missed=self._missed,
                           nearest=near, reason=why)
                self._clear()
                if not cfg.relock_after_lost:
                    self._blocked = True
                    return None
                return self._acquired(self._acquire(cands, diag), entities)
            if self._missed == 1:
                self._emit(LockEventKind.COAST_START, prev=prev, nearest=near, reason=why)
            return prev                              # coasting

        if k > 0:
            self._emit(LockEventKind.REACQUIRED, prev=prev, new=current, missed=k)

        cur_id = current.track_id
        if cfg.jump_heights > 0 and not (self._known_id is not None and cur_id == self._known_id):
            disp = math.hypot(current.poi[0] - anchor[0], current.poi[1] - anchor[1])
            if disp > cfg.jump_heights * max(prev.height, 1.0):
                self._emit(LockEventKind.JUMP, prev=prev, new=current, nearest=disp)
        rebound = current.label != prev.label or (
            cur_id is not None and self._known_id is not None and cur_id != self._known_id)
        if rebound:
            self._emit(LockEventKind.REBOUND, prev=prev, new=current)
            self._vel = None                         # скачок опорной точки - не скорость цели
        elif elapsed > 0:                            # нулевой dt (тот же t дважды) - скорость не трогаем
            # собственная скорость цели: наблюдаемое смещение МИНУС сдвиг от нашей камеры
            vx = (current.poi[0] - prev.poi[0] - acc[0]) / elapsed
            vy = (current.poi[1] - prev.poi[1] - acc[1]) / elapsed
            old = self._vel
            self._vel = (vx, vy) if old is None else (0.5 * (old[0] + vx), 0.5 * (old[1] + vy))
        if cur_id is not None:
            self._known_id = cur_id                  # None не затирает известный ID

        self._missed = 0
        self._ego_acc = (0.0, 0.0)
        self._t_obs = t
        skip_ov = cfg.skip_overlapping_challengers
        cb = current.bbox
        cpx, cpy = current.poi
        ch = None
        for e in cands:                              # лучший претендент без создания списка
            if e is current:
                continue
            if skip_ov:                              # перекрывает цель - это, скорее всего, она же
                ex, ey = e.poi
                eb = e.bbox
                if ((cb[0] <= ex <= cb[2] and cb[1] <= ey <= cb[3])
                        or (eb[0] <= cpx <= eb[2] and eb[1] <= cpy <= eb[3])):
                    continue
            if ch is None or self._acquire_key(e) < self._acquire_key(ch):
                ch = e
        if ch is not None and not cfg.hard_lock:
            better = (ch.aimed and not current.aimed) or (
                ch.aimed == current.aimed
                and ch.distance_px < current.distance_px * (1.0 - cfg.switch_margin))
            self._challenger_streak = self._challenger_streak + 1 if better else 0
            if self._challenger_streak >= cfg.switch_frames:
                self._emit(LockEventKind.SWITCHED, prev=current, new=ch)
                current, self._challenger_streak = ch, 0
                self._known_id = ch.track_id
                self._vel = None
        else:
            self._challenger_streak = 0

        self._target = current
        self._sep = self._separation(current, entities)
        return current

    def _acquired(self, target: Optional[TrackedEntity],
                  entities: Sequence[TrackedEntity]) -> Optional[TrackedEntity]:
        if target is not None:
            self._sep = self._separation(target, entities)
        return target

    @staticmethod
    def _separation(cur: TrackedEntity, entities: Sequence[TrackedEntity]) -> float:
        """Расстояние от цели до ближайшей ДРУГОЙ сущности, px. Перекрывающие цель (голова своего
        же тела, дубль бокса) соседями не считаются."""
        cb = cur.bbox
        cx, cy = cur.poi
        best = math.inf
        for e in entities:
            if e is cur:
                continue
            ex, ey = e.poi
            eb = e.bbox
            if ((cb[0] <= ex <= cb[2] and cb[1] <= ey <= cb[3])
                    or (eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3])):
                continue
            d = math.hypot(ex - cx, ey - cy)
            if d < best:
                best = d
        return best


def ego_shift_from_mouse(move_x: float, move_y: float,
                         px_per_unit: float = 1.0) -> tuple[float, float]:
    """Сдвиг неподвижного мира на экране от вашего движения мыши/камеры.

    Мышь вправо - камера вправо - мир уезжает ВЛЕВО, поэтому знак минус:
    shift = -move * px_per_unit. px_per_unit - во сколько пикселей экрана
    превращается единица вашего move (зависит от чувствительности игры; оценить
    можно по estimate_ego_scale)."""
    return -move_x * px_per_unit, -move_y * px_per_unit


def estimate_ego_scale(pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> float:
    """Оценка px_per_unit методом наименьших квадратов.

    pairs - [((move_x, move_y), (наблюдаемый_сдвиг_x, наблюдаемый_сдвиг_y)), ...] для
    НЕПОДВИЖНОГО объекта (или усреднения по множеству кадров). Модель: shift = -scale * move.
    Возвращает scale (0.0, если движений не было)."""
    num = den = 0.0
    for (mx, my), (ox, oy) in pairs:
        num += mx * ox + my * oy
        den += mx * mx + my * my
    return -num / den if den > 0 else 0.0


class EgoTracker:
    """Собирает ego_shift для замка из ВАШЕГО наведения и джиттера.

    Каждый кадр:  shift = ego.update(aim_move=<move, отправленный с прошлого кадра>)
                  life.update(entities, size, held, shift)

    Джиттер двигает камеру в отдельном потоке (сотни раз в секунду), и замок не знает его фазу
    в момент съёмки кадра. Передайте jitter_offset=jitter.offset_at (JitterController): трекер
    возьмёт СМЕЩЕНИЕ джиттера на момент кадра (t - latency_s) и вычтет предыдущее - получится
    сдвиг джиттера именно между двумя кадрами. Он складывается с aim_move и переводится в пиксели
    (ego_shift_from_mouse). Без jitter_offset учитывается только aim_move.

    px_per_unit - пикселей экрана на единицу мыши (см. estimate_ego_scale).
    latency_s - задержка от движения до появления его на кадре (захват + передача): подберите так,
    чтобы nearest_px в COASTING после рывков был минимальным.
    """

    def __init__(self, px_per_unit: float = 1.0,
                 jitter_offset: Optional[Callable[[float], tuple[float, float]]] = None,
                 latency_s: float = 0.0,
                 clock: Callable[[], float] = time.perf_counter,
                 aim_latency_s: float = 0.0, aim_smear_s: float = 0.0) -> None:
        """aim_latency_s / aim_smear_s (v2.12): отклик картинки на ВАШЕ наведение запаздывает и растянут
        во времени (отправка движения идёт потоком, плюс захват/NDI). По 894 кадрам лога 06.11 лучше
        всего подошли aim_latency_s=0.03, aim_smear_s=0.04 при px_per_unit=0.7 (корреляция сдвига
        бокса с движением мыши 0.43 вместо 0.27 при мгновенном применении). Движение, отправленное
        между кадрами, считается приходящим равномерно на интервале
        [t_движения + aim_latency_s, + aim_smear_s]. Оба 0 (по умолчанию) = прежнее поведение:
        весь aim_move применяется на ближайшем кадре. Джиттер от этих параметров не зависит
        (для него latency_s)."""
        if not math.isfinite(px_per_unit):
            raise ValueError(f"px_per_unit must be finite, got {px_per_unit!r}")
        if not (math.isfinite(latency_s) and latency_s >= 0):
            raise ValueError(f"latency_s must be >= 0, got {latency_s!r}")
        for name, v in (("aim_latency_s", aim_latency_s), ("aim_smear_s", aim_smear_s)):
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be >= 0, got {v!r}")
        self.px_per_unit = px_per_unit
        self.latency_s = latency_s
        self.aim_latency_s = aim_latency_s
        self.aim_smear_s = aim_smear_s
        self._jitter = jitter_offset
        self._clock = clock
        self._last: Optional[tuple[float, float]] = None
        self._t_prev: Optional[float] = None
        # [t_движения, mx, my]: движения, чей отклик ещё не пришёл целиком
        self._pending: list[list[float]] = []

    def set_jitter_offset(self, jitter_offset: Optional[Callable[[float], tuple[float, float]]]) -> None:
        """Подключить (или отключить) джиттер ПОСЛЕ создания трекера: контроллер часто создают позже
        (например, в start()). Передайте jitter.offset_at; None - не учитывать джиттер."""
        self._jitter = jitter_offset
        self._last = None

    def reset(self) -> None:
        """Забыть прошлое смещение джиттера и ещё не пришедший отклик на наведение."""
        self._last = None
        self._t_prev = None
        self._pending.clear()

    def _aim_arrived(self, mx: float, my: float, t: float) -> tuple[float, float]:
        """Сколько единиц наведения ДОШЛО до картинки между прошлым и этим кадром."""
        t_prev = t if self._t_prev is None else self._t_prev
        self._t_prev = t
        if mx or my:
            self._pending.append([0.5 * (t_prev + t), mx, my])
        lat, sm = self.aim_latency_s, self.aim_smear_s

        def frac(z: float) -> float:
            if sm <= 1e-9:
                return 1.0 if z >= 0 else 0.0
            return 0.0 if z <= 0 else 1.0 if z >= sm else z / sm

        ax = ay = 0.0
        keep: list[list[float]] = []
        for e in self._pending:
            d = frac(t - e[0] - lat) - frac(t_prev - e[0] - lat)
            ax += e[1] * d
            ay += e[2] * d
            if t - e[0] - lat < sm or (sm <= 1e-9 and t - e[0] - lat < 0):
                keep.append(e)
        self._pending = keep
        return ax, ay

    def update(self, aim_move: tuple[float, float] = (0.0, 0.0),
               t: Optional[float] = None) -> tuple[float, float]:
        """ego_shift (пиксели) для этого кадра. t - время съёмки кадра (perf_counter), по умолчанию now."""
        mx, my = aim_move
        if not (math.isfinite(mx) and math.isfinite(my)):
            mx = my = 0.0
        tt0 = self._clock() if t is None else t
        if self.aim_latency_s > 0 or self.aim_smear_s > 0:
            mx, my = self._aim_arrived(mx, my, tt0)
        jx = jy = 0.0
        if self._jitter is not None:
            tt = tt0 - self.latency_s
            off = self._jitter(tt)
            if math.isfinite(off[0]) and math.isfinite(off[1]):
                if self._last is not None:
                    jx, jy = off[0] - self._last[0], off[1] - self._last[1]
                self._last = (off[0], off[1])
        return ego_shift_from_mouse(mx + jx, my + jy, self.px_per_unit)


# ============================================================================
# Кнопка удержания: дебаунс + диагностика
# ============================================================================
@dataclass(frozen=True, slots=True)
class HoldTransition:
    kind: str                      # "press" | "release"
    frame: int                     # номер кадра (с 1)
    t: float                       # clock() в момент перехода
    held_frames: int = 0           # release: сколько кадров реально держали
    held_s: float = 0.0            # release: длительность, сек
    bridged_dropouts: int = 0      # release: сколько провалов опроса было "залатано" внутри


class ButtonDebounce:
    """Опрос кнопки с защитой от провалов: отпущенной считается, только если
    raw=False держится release_frames кадров подряд (release_frames=1 - без защиты).

    Счётчики для диагностики дребезга: bridged_total - сколько коротких
    провалов (< release_frames) было скрыто; presses - число нажатий.
    Если bridged_total растёт, а пользователь кнопку не отпускал - это дребезг опроса.
    """

    def __init__(self, release_frames: int = 3, always_on: bool = False,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        if release_frames < 1:
            raise ValueError("release_frames must be >= 1")
        self.release_frames = release_frames
        self.always_on = always_on
        self._clock = clock
        self._frame = 0
        self._held = False
        self._streak = 0               # подряд кадров с raw=False
        self._press_frame = 0
        self._press_t = 0.0
        self._dropouts = 0             # залатанные провалы в текущем удержании
        self.bridged_total = 0
        self.presses = 0
        self._transition: Optional[HoldTransition] = None

    @property
    def held(self) -> bool:
        return self._held

    @property
    def transition(self) -> Optional[HoldTransition]:
        """Переход, случившийся на последнем update(), иначе None."""
        return self._transition

    def update(self, raw_held: bool) -> bool:
        self._frame += 1
        self._transition = None
        if raw_held or self.always_on:
            if self._held and self._streak > 0:
                self._dropouts += 1
                self.bridged_total += 1
            self._streak = 0
            if not self._held:
                self._held = True
                self._press_frame, self._press_t, self._dropouts = self._frame, self._clock(), 0
                self.presses += 1
                self._transition = HoldTransition("press", self._frame, self._press_t)
        else:
            self._streak += 1
            if self._held and self._streak >= self.release_frames:
                now = self._clock()
                last_true = self._frame - self._streak
                self._transition = HoldTransition(
                    "release", self._frame, now,
                    held_frames=last_true - self._press_frame + 1,
                    held_s=now - self._press_t, bridged_dropouts=self._dropouts)
                self._held = False
        return self._held


# ============================================================================
# Жизненный цикл: кнопка + замок + понятные логи причин
# ============================================================================
@dataclass(frozen=True, slots=True)
class LifecycleResult:
    target: Optional[TrackedEntity]
    status: LockStatus
    held: bool
    events: tuple[LockEvent, ...]
    transition: Optional[HoldTransition]


def _fmt_entity(e: Optional[TrackedEntity]) -> str:
    if e is None:
        return "-"
    return f"id={e.track_id} {e.label} conf={e.confidence:.2f} dist={e.distance_px:.1f}px"


def _fmt_nearest(ev: LockEvent) -> str:
    """Была ли цель в кадре: маленькое расстояние = видна, но не узнана."""
    if ev.nearest_px is None:
        return "в кадре никого"
    why = f" | не цель: {ev.reason}" if ev.reason else ""
    return f"ближайшая видимая сущность в {ev.nearest_px:.1f}px от ожидаемой позиции{why}"


class TargetLifecycle:
    """Связывает ButtonDebounce и TargetLock и пишет причину КАЖДОГО события.

    Сообщения лога:
      HOLD PRESS / HOLD RELEASE   - переходы кнопки (кадр, длительность, залатанные провалы);
                                    INFO только если за удержание была цель, иначе DEBUG
                                    (log_idle_holds=True - всегда INFO);
      SHORT HOLD                  - удержание с целью короче short_hold_frames;
      TARGET LOCKED / SWITCHED    - захват и реальное переключение (id a->b);
      TARGET REBOUND              - та же цель сменила метку/ID (пометка, если conf < acquire_conf);
      TARGET COASTING / REACQUIRED- временная потеря детекции и возврат;
      RELEASED                    - пользователь отпустил кнопку (missed_frames - реальный, до сброса);
      TARGET LOST (track broken)  - цель пропала при зажатой кнопке дольше lost_frames.
    """

    def __init__(self, lock: TargetLock, debounce: ButtonDebounce, *,
                 priority_only: bool = True, short_hold_frames: int = 3,
                 log_idle_holds: bool = False,
                 logger: Optional[logging.Logger] = None) -> None:
        self.lock = lock
        self.debounce = debounce
        self.priority_only = priority_only
        self.short_hold_frames = short_hold_frames
        self.log_idle_holds = log_idle_holds
        self.log = logger or logging.getLogger("detection_v2.lifecycle")
        self._hold_had_target = False

    def update(self, entities: Sequence[TrackedEntity], frame_size: tuple[int, int],
               raw_held: bool, ego_shift: tuple[float, float] = (0.0, 0.0),
               t: Optional[float] = None) -> LifecycleResult:
        """t - время съёмки кадра (perf_counter), см. TargetLock.update."""
        held = self.debounce.update(raw_held)
        tr = self.debounce.transition
        log = self.log

        if tr is not None and tr.kind == "press":
            self._hold_had_target = False
            log.log(logging.INFO if self.log_idle_holds else logging.DEBUG,
                    "HOLD PRESS | frame=%d", tr.frame)

        if held:
            pool = ([e for e in entities if e.category is Category.PRIORITY]
                    if self.priority_only else entities)
            target = self.lock.update(pool, frame_size, ego_shift, t)
            if target is not None:
                self._hold_had_target = True
        else:
            target = None
            self.lock.reset()                        # RESET-событие с реальным missed_frames

        events = self.lock.events
        for ev in events:
            self._log_event(ev, tr)

        if tr is not None and tr.kind == "release":
            level = logging.INFO if (self._hold_had_target or self.log_idle_holds) else logging.DEBUG
            log.log(level, "HOLD RELEASE | frame=%d held_frames=%d held=%.0fms bridged_dropouts=%d",
                    tr.frame, tr.held_frames, tr.held_s * 1000, tr.bridged_dropouts)
            if self._hold_had_target and tr.held_frames < self.short_hold_frames:
                log.warning("SHORT HOLD | held_frames=%d < %d (дребезг опроса или тап?)",
                            tr.held_frames, self.short_hold_frames)
        return LifecycleResult(target, self.lock.status, held, events, tr)

    def _log_event(self, ev: LockEvent, tr: Optional[HoldTransition]) -> None:
        k = LockEventKind
        if ev.kind is k.ACQUIRED:
            self.log.info("TARGET LOCKED | %s", _fmt_entity(ev.new))
        elif ev.kind is k.SWITCHED:
            self.log.info("TARGET SWITCHED | %s -> %s", _fmt_entity(ev.prev), _fmt_entity(ev.new))
        elif ev.kind is k.REBOUND:
            low = ev.new.confidence < self.lock.config.acquire_conf
            self.log.info("TARGET REBOUND | %s -> %s%s", _fmt_entity(ev.prev), _fmt_entity(ev.new),
                          " | conf ниже acquire_conf (удержание мягкое)" if low else "")
        elif ev.kind is k.COAST_START:
            self.log.info("TARGET COASTING | %s | %s", _fmt_entity(ev.prev), _fmt_nearest(ev))
        elif ev.kind is k.REACQUIRED:
            self.log.info("TARGET REACQUIRED | after %d frames | %s", ev.missed_frames,
                          _fmt_entity(ev.new))
        elif ev.kind is k.LOST_TIMEOUT:
            cfg = self.lock.config
            limit = (f"lost_s={cfg.lost_s:.3f}" if self.lock.timed and cfg.lost_s > 0
                     else f"lost_frames={cfg.lost_frames}")
            self.log.info("TARGET LOST (track broken) | missed_frames=%d > %s | %s | %s%s",
                          ev.missed_frames, limit, _fmt_entity(ev.prev), _fmt_nearest(ev),
                          "" if cfg.relock_after_lost else " | новую цель не берём до отпускания кнопки")
        elif ev.kind is k.JUMP:
            h = max(ev.prev.height, 1.0)
            self.log.info("TARGET JUMP | %s -> %s | смещение %.1fpx = %.1f высот цели (сосед?)",
                          _fmt_entity(ev.prev), _fmt_entity(ev.new), ev.nearest_px, ev.nearest_px / h)
        elif ev.kind is k.RESET:
            extra = f" held_frames={tr.held_frames}" if tr is not None and tr.kind == "release" else ""
            self.log.info("RELEASED | missed_frames=%d%s | %s", ev.missed_frames, extra,
                          _fmt_entity(ev.prev))


# ============================================================================
# Debug overlay (отделён от логики)
# ============================================================================
def draw_debug_overlay(frame: np.ndarray, result: DetectionResult) -> None:
    """Рисует на кадре НА МЕСТЕ (in-place), ничего не возвращает. Требует OpenCV."""
    import cv2  # локальный импорт: ядро работает без OpenCV

    if frame is None:
        return
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def clip(v: float, hi: int) -> int:
        return int(min(max(round(v), 0), hi))

    def put_label(text: str, x: int, y: int, color, scale: float) -> None:
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        x = max(min(x, fw - tw - 2), 2)
        y = min(max(y, th + 2), fh - 5)
        cv2.rectangle(frame, (max(x - 2, 0), max(y - th - 2, 0)),
                      (min(x + tw + 2, fw - 1), min(y + 2, fh - 1)), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, scale, color, 1, cv2.LINE_AA)

    GREEN, YELLOW, GRAY = (0, 255, 0), (0, 255, 255), (200, 200, 200)

    for e in result.entities:
        prio = e.category is Category.PRIORITY
        color, thick = (GREEN, 3) if prio else (YELLOW, 1)
        x1, y1 = clip(e.bbox[0], fw - 1), clip(e.bbox[1], fh - 1)
        x2, y2 = clip(e.bbox[2], fw - 1), clip(e.bbox[3], fh - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        text = f"{e.label} {e.confidence:.2f}"
        if e.track_id is not None:
            text += f" #{e.track_id}"
        put_label(text, x1, y1 - 5, color, 0.5)

    if result.observed_labels:
        put_label(", ".join(sorted(result.observed_labels)), 10, fh - 20, GRAY, 0.4)
    if result.tracking_lost:
        put_label("TRACKING LOST", 10, 20, (0, 0, 255), 0.5)


def describe_config(*configs: Any, ego: Optional["EgoTracker"] = None) -> str:
    """Одна строка для старт-лога: версия модуля и ВСЕ поля переданных dataclass-конфигов
    (LockConfig, DetectorConfig, ...) плюс параметры EgoTracker. Нужна, чтобы по логу было видно,
    с какими настройками шёл прогон (в самих логах версия и конфиг не пишутся)."""
    import dataclasses
    parts = [f"detection_v2 v{__version__}"]
    for cfg in configs:
        if cfg is None:
            continue
        if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
            body = ", ".join(f"{f.name}={getattr(cfg, f.name)!r}" for f in dataclasses.fields(cfg))
        else:
            body = repr(cfg)
        parts.append(f"{type(cfg).__name__}({body})")
    if ego is not None:
        parts.append(f"EgoTracker(px_per_unit={ego.px_per_unit!r}, latency_s={ego.latency_s!r}, "
                     f"aim_latency_s={ego.aim_latency_s!r}, aim_smear_s={ego.aim_smear_s!r}, "
                     f"jitter={'on' if ego._jitter is not None else 'off'})")
    return " | ".join(parts)


def log_startup_config(*configs: Any, ego: Optional["EgoTracker"] = None,
                       logger: Optional[logging.Logger] = None) -> str:
    """Пишет describe_config(...) в лог уровнем INFO (вызвать один раз при старте) и возвращает строку."""
    line = "CONFIG | " + describe_config(*configs, ego=ego)
    (logger or logging.getLogger("detection_v2")).info(line)
    return line
