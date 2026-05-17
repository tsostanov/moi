# Лабораторная работа №5 — Билатеральная фильтрация с сохранением границ

## Что реализовано

| Пункт задания | Файл / функция |
|---|---|
| Рендер с AOV-каналами (direct, indirect, depth, obj_id, normal) | `render_aov.py` |
| Билатеральный фильтр (арифметическое среднее) | `bilateral.py :: bilateral_mean` |
| Билатеральный фильтр (медианный, сложный вариант) | `bilateral.py :: bilateral_median` |
| Нормировка ядра `Σ G_s = 1` | `bilateral.py :: make_spatial_kernel` |
| Edge-stop: obj_id (жёсткий), нормаль (мягкий), глубина (мягкий) | `bilateral.py :: bilateral_mean/median` |
| Раздельная фильтрация direct + indirect | `bilateral.py :: bilateral_filter(split_direct_indirect=True)` |
| Энергетическая нормировка по объектам | `bilateral.py :: energy_normalize` |
| CLI с параметрами и метриками PSNR/L1 | `main.py` |
| Визуализация AOV-каналов (--dump-debug) | `main.py :: debug_obj_id/normal/depth` |
| 8 наборов юнит-тестов (pytest) | `tests/` |

## Структура папки

```
05/
  task.pdf
  PLAN.md
  README.md
  render_aov.py        # расширенный path tracer → AOV (NPZ + preview PNG)
  bilateral.py         # mean / median билатеральный фильтр
  main.py              # CLI: --aov ... --mode mean|median ...
  tests/
    __init__.py
    test_spatial_kernel.py
    test_bilateral_preserves_constant.py
    test_object_edge_preservation.py
    test_normal_edge_preservation.py
    test_energy_conservation.py
    test_noise_reduction.py
    test_aov_io.py
    test_cli_smoke.py
  outputs/
    aov.npz            # шумный рендер (4 spp)
    aov.png            # превью шумного рендера
    reference.npz      # эталон (32 spp)
    reference.png      # превью эталона
    noisy.png          # шумный color (для сравнения)
    filtered_mean.png  # результат mean-фильтра
    filtered_mean.ppm
    filtered_mean.txt  # параметры + метрики PSNR/L1
    filtered_median.png
    filtered_median.ppm
    filtered_median.txt
    diff.png           # |noisy − filtered| * 5
    debug_objid.png    # object id в палитре
    debug_normal.png   # нормаль как (n+1)/2
    debug_depth.png    # глубина нормированная
```

## Как запускать

### 1. Сгенерировать шумный кадр + AOV (4 spp)
```bash
python 05/render_aov.py --width 500 --height 500 --samples 4 \
    --max-depth 5 --output 05/outputs/aov.npz
```

### 2. Сгенерировать эталон (32+ spp)
```bash
python 05/render_aov.py --width 500 --height 500 --samples 32 \
    --max-depth 6 --output 05/outputs/reference.npz
```

### 3. Mean-фильтр с раздельной фильтрацией direct+indirect
```bash
python 05/main.py --aov 05/outputs/aov.npz \
    --mode mean --sigma-s 3 --sigma-n 0.3 --sigma-z 0.1 \
    --split-direct-indirect \
    --energy-normalize object \
    --reference 05/outputs/reference.npz \
    --output 05/outputs/filtered_mean.png
```

### 4. Медианный фильтр
```bash
python 05/main.py --aov 05/outputs/aov.npz \
    --mode median --radius 3 \
    --energy-normalize object \
    --reference 05/outputs/reference.npz \
    --output 05/outputs/filtered_median.png
```

### 5. Визуализация AOV-каналов
```bash
python 05/main.py --aov 05/outputs/aov.npz --mode mean \
    --output 05/outputs/filtered_mean.png --dump-debug
```
Создаёт: `debug_objid.png`, `debug_normal.png`, `debug_depth.png`, `noisy.png`, `diff.png`.

### 6. Юнит-тесты (38 тестов)
```bash
pytest 05/tests -q
```

---

## Формула фильтра (соответствие слайду 45)

```
g_p = (1/W_p) * Σ_{q∈S} f_q · G_s(p−q) · G_r(p,q)
```

`G_s` — дискретное Гауссово ядро, нормированное так, что `Σ G_s = 1`.  
Тогда `W_p = Σ G_r(p,q)` — нормировка только по диапазонным весам.

### Компоненты диапазонного веса G_r

| Компонент | Формула |
|---|---|
| `w_obj` | `1` если `obj_id_p == obj_id_q`, иначе `0` (жёсткая маска) |
| `w_normal` | `exp(-(1 - max(0, n_p·n_q)) / σ_n²)` |
| `w_depth` | `exp(-(z_p - z_q)² / (2·σ_z²))` |
| `G_r` | `w_obj · w_normal · w_depth` |

---

## Параметры по умолчанию (Корнельский ящик, 500×500)

| Параметр | Значение | Обоснование |
|---|---|---|
| `sigma_s` | 3.0 | ~3 пикселя → сглаживает шум зерна SPP=4 |
| `sigma_n` | 0.3 | терпим к нормалям ±30°, отсекает >90° |
| `sigma_z` | 0.1 | малый σ → разделяет объекты на ~0.1 единицу |
| `radius` | 3 | окно 7×7 — баланс скорости и качества |

---

## Что говорить на защите

1. **Зачем раздельно фильтровать direct и indirect?**  
   Компоненты независимы и имеют разную статистику шума: direct — жёсткие тени (единица либо 0), indirect — мягкий шум от диффузных отражений. Раздельная фильтрация сохраняет корректный характер шума в каждом канале.

2. **Почему нормировать ядро `G_s` к сумме 1?**  
   Это ровно условие со слайда 45: `Σ G_s = 1` упрощает знаменатель до `W_p = Σ G_r`, что позволяет считать нормировку за один проход.

3. **Почему obj_id — жёсткая маска, а не ещё один Гаусс?**  
   Смешивание разных объектов всегда физически некорректно (разная материальная модель). Жёсткая маска гарантирует, что фоновые пиксели не «вытекают» на объекты.

4. **Что делает энергетическая нормировка?**  
   Билатеральный mean сохраняет яркость локально (за счёт деления на `W_p`), но не глобально. После нормировки `Σ_{p∈O} g_p = Σ_{p∈O} f_p` для каждого объекта O.

5. **Медианный vs. mean?**  
   Mean лучше сохраняет детали на плавных поверхностях (весовое усреднение). Median устойчив к выбросам (яркие пятна от одного теневого луча) и не создаёт артефактов «гало».
